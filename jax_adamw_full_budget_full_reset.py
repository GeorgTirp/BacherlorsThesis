"""Submission file for an NAdamW optimizer with warmup+cosine LR in Jax."""
import inspect
import functools
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import csv
# isort: off
# We have to turn off isort here to resolve a conflict between isort and yapf.
from typing import (Any,
                    Callable,
                    Dict,
                    Iterator,
                    List,
                    NamedTuple,
                    Optional,
                    Tuple,
                    Union)
# isort: on

import chex
from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec

_GRAD_CLIP_EPS = 1e-6

#####variables just to test
GRAPH = {"make_graph": True, "counter": 0}
#####


#Params for multiple learing rate schedules (self adjusting according to chosen schedules)
SCHEDULE_PARAMS = {
  "has_multiple": False,  #self adjusting
  "reinit_steps": None,   #self adjusting
  "warmup_schedule_fns": [], #have to be set later within init_optimizer_state from line 324
  "decay_schedule_fns":[],   #have to be set later within init_optimizer_state from line 324
  "alter_lr": [0, 0.002, -0.001],
  "alter_warmupfactor": [0, 0, 0],
  "num_early_stops": 0,
  "currently_stopping": False,
  "stop_metric": {"evaluate" :False, "metrics": [], "global_min": [0, np.inf], "last_loss": None},
  "reset_metric": {"metric":0, "updates": None, "params": None}
}

#HPARAMS from github

HPARAMS = {
    "dropout_rate": 0.1,
    "learning_rate": 0.0017486387539278373,
    "one_minus_beta1": 0.06733926164,
    "beta2": 0.9955159689799007,
    "weight_decay": 0.08121616522670176,
    "warmup_factor": 0.02
}



# Forked from
# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/alias.py
def adamw(
    learning_rate: Union[float, optax.Schedule],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://opq enreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the official PyTorch
  implementation also follows this).
  Current code implements a simpler version with no momentum decay and slightly
  different bias correction terms. The exact description can be found here
  https://arxiv.org/pdf/1910.05446.pdf (Table 1).

  Args:
    learning_rate: A fixed global scaling factor.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: Wheter to use bias correction.
    weight_decay:h Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Nadam gradient transformations are applied to all parameters.

  Returns:
    An (init_fn, update_fn) tuple.
  """
  return optax.chain(
      scale_by_adam(b1, b2, eps, eps_root, debias),
      optax.add_decayed_weights(weight_decay, weight_decay_mask),
      scale_by_learning_rate(learning_rate))


# All functions below are forked from
# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/transform.py
def scale_by_adam(b1: float = 0.9,
                   b2: float = 0.999,
                   eps: float = 1e-8,
                   eps_root: float = 0.0,
                   debias: bool = True,
                   power: float = 0.5) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the pytorch imp. also
  follows this).

  Current code implements a simpler version with no momentum decay and slightly
  different (standard Adam) bias correction terms. The exact description can be
  found here https://arxiv.org/pdf/1910.05446.pdf (Table 1)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: Whether to use bias correction.
    power: The power to use in the preconditioner (0.5 in default adam).
  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params): 
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    #mu_hat = _update_moment(updates, mu, b1, 1)
    mu_hat = mu_hat if not debias else _bias_correction(mu, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)
  
  return optax.GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(NamedTuple):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates
  nu: optax.Updates




def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState,
                         optimizer_state: spec.OptimizerState = None,
                         schedule_params = SCHEDULE_PARAMS) -> spec.OptimizerState:
  """Creates a AdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state   
  del rng
  del hyperparameters

  #Full budget has 3 times the stepsize than a full algoperf run
  #workload.step_hint = workload.step_hint * 2 
  hyperparameters = SimpleNamespace(**HPARAMS)

  #Define warm up strategies
  def jax_linear_warmup(step_hint: int, hyperparameters, alter_lr_by=0, alter_warmup_by = 0):
        # Create linear learning rate  warm up funtion
    warmup_steps = warmup_steps = int((hyperparameters.warmup_factor + alter_warmup_by) * step_hint)
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=hyperparameters.learning_rate + alter_lr_by,
        transition_steps=warmup_steps)
    return warmup_fn
  
  def jax_cosine_warmup(step_hint: int, hyperparameters, alter_lr_by=1, alter_warmup_by = 1):
    # Create cosine learning rate  warm up funtion
    pass

  def jax_exponential_warmup(step_hint: int, hyperparameters, alter_lr_by=1, alter_warmup_by = 1):
    # Create exponential learning rate  warm up funtion
    pass

  
  #Define learing rate schedules
  def jax_cosine_decay(step_hint: int, hyperparameters, alter_lr_by=0, alter_warmup_by = 0):
    warmup_steps = int((alter_warmup_by + hyperparameters.warmup_factor) * step_hint)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=alter_lr_by + hyperparameters.learning_rate, decay_steps=cosine_steps)
    return cosine_fn

  def jax_exponential_decay(step_hint: int, hyperparameters, alter_lr_by=0, alter_warmup_by = 0):
        # Create learning rate schedule with exponential decay.
    alter_warmup_by
    warmup_steps = int((alter_warmup_by + hyperparameters.warmup_factor) * step_hint)
    exponential_fn = optax.exponential_decay(
      init_value = alter_lr_by + hyperparameters.learning_rate, 
      transition_steps = step_hint-step_hint*0.4, 
      decay_rate = 0.1, 
      transition_begin= warmup_steps + step_hint * 0.2, 
      staircase=False, 
      end_value=None)
    return exponential_fn
  
  def jax_linear_decay(step_hint: int, hyperparameters, alter_lr_by=0, alter_warmup_by = 0):
        # Create learning rate schedule with linear decay.
    warmup_steps = int((alter_warmup_by + hyperparameters.warmup_factor) * step_hint)
    linear_fn = optax.linear_schedule(
      init_value = alter_lr_by + hyperparameters.learning_rate,
      end_value =  0,
      transition_steps = step_hint-warmup_steps*2-step_hint*0.2, 
      transition_begin= warmup_steps + step_hint*0.2)
    return linear_fn
  
  def jax_constant_lr(step_hint, hyperparameters, alter_lr_by=1, alter_warmup_by = 1):
    del step_hint
    del alter_warmup_by
    #Create constant learning rate schedule
    return optax.constant_schedule(alter_lr_by + hyperparameters.learning_rate)

  def combine_warmup_decay_fn(
      step_hint: int,
      hyperparameters, 
      warmup_fn, 
      decay_fn,
      alter_lr_by,
      alter_warmup_by):
    
    warmup_steps = int((hyperparameters.warmup_factor + alter_warmup_by)  * step_hint)
    warmup_fn = warmup_fn(step_hint, hyperparameters, alter_lr_by, alter_warmup_by)
    decay_fn = decay_fn(step_hint, hyperparameters, alter_lr_by, alter_warmup_by)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])
    return schedule_fn
  
  def  jax_lr_schedule(
    #takes multiple schedules and chains them. Also returns info that is used to reinitialize the model
      step_hint: int, 
      hyperparameters, 
      schedule_params):
    
    
    #check if inpues are correct and which glogal settings have to be adjusted for the training loop
    num_schedules = len(schedule_params["decay_schedule_fns"])
    multiple_decay = False if num_schedules <= 1 else True
    if not len(schedule_params["warmup_schedule_fns"])== num_schedules == len(schedule_params["alter_lr"]) == len(schedule_params["alter_warmupfactor"]):
      raise Exception("Number of warmup- and decay schedules and altering lists do not match in Length")

    if (num_schedules - schedule_params["num_early_stops"]) >= 1 and schedule_params["currently_stopping"]:
      len_one_schedule = step_hint // (num_schedules -schedule_params["num_early_stops"])
      boundaries = schedule_params["reinit_steps"]
      #boundaries = list(range(len_one_schedule, step_hint+ len_one_schedule, len_one_schedule))#get boundaries 
      #index = next((i for i, element in enumerate(schedule_params["reinit_steps"]) if element >= boundaries[0]), None)
      #boundaries = schedule_params["reinit_steps"][:index] + boundaries
      currently_stopping = False
    elif schedule_params["reinit_steps"] != None:
      boundaries = schedule_params["reinit_steps"]
      len_one_schedule = schedule_params["reinit_steps"][-1]- schedule_params["reinit_steps"][-2]
    else:
      len_one_schedule = step_hint // num_schedules-schedule_params["num_early_stops"]
      boundaries = list(range(len_one_schedule, step_hint+ len_one_schedule, len_one_schedule))#get boundaries 
    currently_stopping = False

    #zip the warmup and decay schedules togther with altering as one schedule each
    zipped_warmup_and_decay = []
    for i in range(num_schedules):
      zipped_warmup_and_decay.append(combine_warmup_decay_fn(
        len_one_schedule,
        hyperparameters,
        schedule_params["warmup_schedule_fns"][i],
        schedule_params["decay_schedule_fns"][i],
        alter_lr_by = schedule_params["alter_lr"][i],
        alter_warmup_by = schedule_params["alter_warmupfactor"][i]))
    
    #chain all the desired schedules (each warmup+decay) after another
    lr_schedule = optax.join_schedules(
      schedules= zipped_warmup_and_decay,
      boundaries=boundaries)
    
    #Set boundaries for none if not needed later in Training
    boundaries = boundaries if multiple_decay else None
    return lr_schedule, multiple_decay, boundaries, currently_stopping

  #Chose schedule decay  and warmup functions
  schedule_params["decay_schedule_fns"]= [
    jax_cosine_decay,
    jax_cosine_decay,
    jax_cosine_decay]
  schedule_params["warmup_schedule_fns"]= [
    jax_linear_warmup,
    jax_linear_warmup,
    jax_linear_warmup]

  #set budget according to actual runtime
  budget = workload.step_hint *2
  # Create optimizer + LR schedule.
  lr_schedule_fn, schedule_params["has_multiple"], schedule_params["reinit_steps"], schedule_params["currently_stopping"]  = jax_lr_schedule(
    budget,
    hyperparameters,
    schedule_params)
  global  SCHEDULE_PARAMS
  SCHEDULE_PARAMS = schedule_params
  opt_init_fn, opt_update_fn = adamw(
      learning_rate=lr_schedule_fn,
      b1=1.0 - hyperparameters.one_minus_beta1,
      b2=hyperparameters.beta2,
      eps=1e-8,
      weight_decay=hyperparameters.weight_decay)
  
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
  optimizer_state = opt_init_fn(params_zeros_like)



  #Plotting the learning rate schedule 
  global GRAPH
  if GRAPH["make_graph"]:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Bitstream Charter"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif",
    })
    x_values = np.linspace(0, budget, workload.step_hint)
    y_values = lr_schedule_fn(x_values)
    # Create a plot
    plt.plot(x_values, y_values, label='Learning rate', color=(0.7, 0, 0), linewidth=1.5)

    def custom_y_formatter(y, pos):
        # Check for very small values to avoid issues
        if np.abs(y) < 1e-10:
            return '0'  # Return '0' for very small values to avoid log-related errors

        # Determine the order of magnitude of the number
        order_of_magnitude = int(np.floor(np.log10(np.abs(y))))

        # Calculate the scaled value for the desired format (e.g., 1.0e-3, 1.2e-2, etc.)
        scaled_value = y / (10**order_of_magnitude)  # Scale the value to have one digit before the decimal point

        # Format the y-value with the scaled value and the order of magnitude
        formatted_str = f'{scaled_value:.1f}e{order_of_magnitude}'  # Format with one decimal place and scientific notation
        return formatted_str
    
    def custom_x_formatter(x, pos):
        # Check for very small values to avoid issues
        if np.abs(x) < 1e-10:
            return '0'  # Return '0' for very small values to avoid log-related errors

        # Determine the order of magnitude of the number
        order_of_magnitude = int(np.floor(np.log10(np.abs(x))))

        # Calculate the scaled value for the desired format (e.g., 20e3, 40e3, 60e3, etc.)
        scaled_value = x / 1e3  # Divide by 1000 to scale the value for the desired format

        # Format the x-value with the scaled value and the order of magnitude
        formatted_str = f'{scaled_value:.0f}e3'  # Format as integer and scientific notation with 'e3'
        return formatted_str
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(custom_x_formatter))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(custom_y_formatter))
  
    plt.title("Learning Rate Schedule")
    plt.ylabel("Learning Rate")
    plt.xlabel("Global Steps")
    plt.locator_params(axis='x', nbins=10)
    plt.grid(True)
    figname = '/experiment_runs/wmt_lr' + str(GRAPH["counter"])+ '.pdf'
    plt.savefig(figname)
    plt.clf()
    counter = GRAPH["counter"] + 1
    GRAPH.update({"counter": counter})

  return jax_utils.replicate(optimizer_state), opt_update_fn


def model_reinit_schedule(
    global_step: int,
    workload: spec.Workload,
    schedule_params: dict,
    rng: spec.RandomState,
    hyperparameters: spec.Hyperparameters,
    model_state: spec.ModelAuxiliaryState,
    optimizer_state: spec.OptimizerState,
    model_params,
    alpha=0.5):
  
  step_hint = workload.step_hint * 4
  #check for early stopping
  def is_stopping_early(schedule_params: dict, epsilon, epsilon2, recovery_time):
      #define consitions under which the schedule cycle should be stopped early
      if schedule_params["stop_metric"]["evaluate"]:
        v_losses = jnp.array(schedule_params["stop_metric"]["metrics"])
        gradient = jnp.gradient(v_losses)
        current_min_index, current_min = schedule_params["stop_metric"]["global_min"]
        if (len(v_losses) - current_min_index) >= recovery_time \
          and (gradient[-1] * recovery_time + v_losses[-1]) >= current_min+ epsilon2:
          print(f"Case 1 with min:{current_min} and loss: {v_losses[-1]}")
          return True
        elif gradient[-1] > epsilon:
          print(f"Case 2 with gradient:{gradient[-1]}")
          return True
        else: 
          return False
      else: 
        return False


  is_stopping = is_stopping_early(schedule_params, epsilon=0.1, epsilon2= 0.01,recovery_time=9)
  if is_stopping: schedule_params["stop_metric"]["evaluate"] = False
  #Wrapper for implementing early stopping
  def early_stopping(schedule_params: dict):

    num_early_stops = schedule_params["num_early_stops"] + 1
    steps_left = step_hint-global_step
    num_cycles_left = len(schedule_params["alter_lr"]) - num_early_stops
    if num_cycles_left <= 0:
            num_early_stops = schedule_params["num_early_stops"]
            new_reinit_steps = schedule_params["reinit_steps"]
    len_one_schedule = steps_left // num_cycles_left
    boundaries = list(range(len_one_schedule, steps_left+ len_one_schedule, len_one_schedule))#get boundaries 
    new_reinit_steps = [global_step] + [x + global_step for x in boundaries]
    return num_early_stops, new_reinit_steps
  
  if is_stopping:
    schedule_params["num_early_stops"], reinit_steps = early_stopping(schedule_params)
    schedule_params.update({"currently_stopping": True})
    schedule_params.update({"reinit_steps": []})
    for x in reinit_steps:
      schedule_params["reinit_steps"].append(x)

  def is_zero(x):
  #tests if moments got reinit to zero
    return not(jnp.all(x == 0))
  def is_same(x,y):
    return jnp.all(jnp.isclose(x,y,atol=1e-9))
  
 
  #Restart the model on the scheduled timestamps
  if (schedule_params["has_multiple"] and global_step in schedule_params["reinit_steps"]) or is_stopping:
    new_params, _ = workload.init_model_fn(rng, dropout_rate=hyperparameters["dropout_rate"])
    optimizer_state_sec_obj = optimizer_state[2]
    count = optimizer_state[0].count
    reinit_optimizer_state = init_optimizer_state(
      workload,
      model_params,
      model_state,
      hyperparameters,
      rng,
      optimizer_state,
      schedule_params)[0]
    del optimizer_state
    print('Cycle is stopping, reset in progress')
    #with open('/experiment_runs/mult_method_reinits.csv', 'a', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow([global_step, schedule_params['num_early_stops'], GRAPH["counter"]])  # Writing the number as a single-element list
    #with open('/experiment_runs/mult_method_vlosses.csv', 'a', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow(schedule_params['stop_metric']['metrics'])
    #Tests
    #mu = reinit_optimizer_state[0].mu
    #nu = reinit_optimizer_state[0].nu
    #check_mu = jax.tree_util.tree_map(is_zero, mu)
    #mu_is_not_zero = jax.tree_util.tree_all(check_mu)
    #check_nu = jax.tree_util.tree_map(is_zero, nu)
    #nu_is_not_zero = jax.tree_util.tree_all(check_nu)
    #check_model = jax.tree_util.tree_map(is_same, model_params, new_params)
    #model_is_same = jax.tree_util.tree_all(check_model)
    #if mu_is_not_zero or nu_is_not_zero: 
    #  raise Exception("Reinitialization of optimizer failed")
    #if model_is_same:
    #  raise Exception("Reinitialization of model parameters failed")
    
    
    schedule_params["stop_metric"] = {"evaluate" :False, "metrics": [], "global_min": [0, np.inf]}
    with open('/experiment_runs/mult_method_reinits.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([global_step, reset_factor,schedule_params['num_early_stops'], GRAPH["counter"]])  # Writing the number as a single-element list
    with open('/experiment_runs/mult_method_vlosses.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(schedule_params['stop_metric']['metrics'])
    update_sp = True
  else: 
    new_params = model_params
    new_optimizer_state = optimizer_state
    update_sp = False

  return new_params, new_optimizer_state, schedule_params, update_sp


@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 1),
    donate_argnums=(2, 3, 4))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       batch,
                       rng,
                       grad_clip,
                       label_smoothing):

  def _loss_fn(params):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
      current_param_container)
  # Get correct global mean loss and grad.
  (summed_loss, n_valid_examples, grad) = lax.psum(
      (summed_loss, n_valid_examples, grad), axis_name='batch')
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)
  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state, loss, grad_norm


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  #del eval_results
  del hyperparameters

  hyperparameters = HPARAMS

  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters.label_smoothing
  else:
    label_smoothing = 0.0
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None
  outputs = pmapped_train_step(workload,
                               opt_update_fn,
                               model_state,
                               optimizer_state,
                               current_param_container,
                               batch,
                               per_device_rngs,
                               grad_clip,
                               label_smoothing)
  new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs

  global SCHEDULE_PARAMS
  #Applying reinit schedule
  new_params, new_optimizer_state, schedule_params, update_sp = model_reinit_schedule(
    global_step,
    workload,
    SCHEDULE_PARAMS,
    rng,
    hyperparameters,
    new_model_state,
    new_optimizer_state,
    new_params)
  
  
  if update_sp:
    SCHEDULE_PARAMS = schedule_params
  

  if global_step % 100 == 0 and workload.metrics_logger is not None:
    if SCHEDULE_PARAMS["stop_metric"]["metrics"] != [] and eval_results != []:
      current_vloss = eval_results[-1][-1]["validation/loss"]
      last_vloss = SCHEDULE_PARAMS["stop_metric"]["last_loss"]
      if current_vloss != last_vloss:
        SCHEDULE_PARAMS["stop_metric"]["last_loss"] = current_vloss
        SCHEDULE_PARAMS["stop_metric"]["evaluate"] = True
        target_metric = eval_results[-1][-1]["validation/" + workload.target_metric_name]
        SCHEDULE_PARAMS["reset_metric"]["metric"] = target_metric
        print(f'current target metric:{SCHEDULE_PARAMS["reset_metric"]["metric"]}')
        last_smoothed_vloss = SCHEDULE_PARAMS["stop_metric"]["metrics"][-1]
        smoothed_vloss = 0.5 * current_vloss + 0.5 * last_smoothed_vloss
        print(f'current smoothed loss: {smoothed_vloss}')
        SCHEDULE_PARAMS["stop_metric"]["metrics"].append(smoothed_vloss)
        if smoothed_vloss <= SCHEDULE_PARAMS["stop_metric"]["global_min"][1]:
          SCHEDULE_PARAMS["stop_metric"]["global_min"] = [len(SCHEDULE_PARAMS["stop_metric"]["metrics"]), smoothed_vloss]
      else:
        SCHEDULE_PARAMS["stop_metric"]["evaluate"] = False
    elif SCHEDULE_PARAMS["stop_metric"]["metrics"] == [] and eval_results != []: 
      current_vloss = eval_results[-1][-1]["validation/loss"]
      SCHEDULE_PARAMS["stop_metric"]["last_loss"] = current_vloss
      SCHEDULE_PARAMS["stop_metric"]["metrics"].append(current_vloss)

    # Log loss, grad_norm.
    #workload.metrics_logger
    workload.metrics_logger.append_scalar_metrics(
        {
            'loss': loss[0],
            'grad_norm': grad_norm[0],
        }, global_step)
  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 131_072
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 64
  elif workload_name == 'mnist':
    return 16
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch

