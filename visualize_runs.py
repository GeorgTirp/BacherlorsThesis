import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib.ticker import ScalarFormatter
import pickle

def reset_steps(filename):
    numbers = []
    # Open the CSV file in read mode
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each row (which is a list) to an integer and append to the list
            numbers.append(int(row[0]))
    return numbers

def adjust_vector_length(x, y):
    """
    Adjusts the length of vector y to match the length of vector x.
    If y is longer than x, it is truncated from the end.
    If y is shorter than x, it is padded with the last value of y.
    
    Args:
        x (np.ndarray): Vector x.
        y (np.ndarray): Vector y.
    
    Returns:
        np.ndarray: Adjusted vector y.
    """
    if len(y) > len(x):
        return y[:len(x)]
    elif len(y) < len(x):
        last_value = y[-1] if len(y) > 0 else 0
        padding_length = len(x) - len(y)
        return np.concatenate((y, np.full(padding_length, last_value)))
    else:
        return y

def round_up_to_even(number):
    # Find the order of magnitude
    order_of_magnitude = 10 ** (math.floor(math.log10(number)) + 1)

    # Round up to the next first even digit
    rounded_number = math.ceil(number / order_of_magnitude) * order_of_magnitude

    return rounded_number

#List of desired measurements to draw a graph to
measures_of_interest = [
    #"accumulated_eval_time",
    #"accumulated_logging_time",
    #"accumulated_submission_time",
    "global_step",
    #"preemption_count",
    #"score",
    #"test/accuracy",
    "test/loss",
    "loss",
    #"test/num_examples",
    #"total_duration",
    #"train/accuracy",
    "train/loss",
    "validation/accuracy",
    "validation/loss",
    #"validation/num_examples"
    "validation/mean_average_precision",
    "validation/bleu"
]
def custom_formatter(x, pos):
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

#read in a experiment validation file
def read_csv_to_dict(file_paths):
    list_of_dicts = []
    for file_path in file_paths:
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            data_dict = {col: [] for col in csv_reader.fieldnames}

            for row in csv_reader:
                for col in csv_reader.fieldnames:
                    data_dict[col].append(float(row[col]))  # Adjust the type conversion as needed

        list_of_dicts.append({col: np.array(data_dict[col]) for col in data_dict})
    return list_of_dicts

#Plot every entry of the dictionary
def plot_and_save_as_pdf(list_of_dicts, target_list, saveing_destination, list_of_labels, list_of_colors, resets):
    max_global_step = int(list_of_dicts[0]["global_step"][-1])
    print(max_global_step)
    num_graphs = len(list_of_dicts)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Bitstream Charter"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif",
    })

    #ticks = round_up_to_even(max_global_step) / 8
    for key, values in list_of_dicts[0].items():
        plt.figure()
        if key in target_list: 
            for i in range(num_graphs):
                vertical_lines = True
                values = list_of_dicts[i][key]
                end = len(values)
                #x_values = np.array(list_of_dicts[0]["score"])
               # end = int(x_values[-1])
                y_values = values
                x_original = np.linspace(0, end, end)
                x_values = np.linspace(0, max_global_step, end)
                y_values = np.repeat(values, len(x_values) // len(x_original))
                
                #y_values = adjust_vector_length(x_values, y_values)
                plt.plot(x_values, y_values, label=list_of_labels[i], linewidth=1, color=list_of_colors[i])

                # Customize x-axis tick labels with ScalarFormatter
                # Manually format x-axis tick labels in scientific notation
                

                plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
                # Plot vertical lines at specified x-values
                            # Define x-values where you want vertical lines
                
                if vertical_lines:
                    vertical_lines = resets
                    text_pos = np.max(y_values / 2)
                    for vline in vertical_lines:
                        plt.axvline(x=vline, color=(0.5, 0.0, 0.0), alpha=0.6 , linewidth=1.2, linestyle='--')
                        plt.text(vline, text_pos, f'Reinit', color='black', fontsize=7, ha='center', va='center', rotation=0,
                    bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.2'))
                    vertical_lines=False

                ## Customize plot appearance
                plt.grid(True)
                title = key.replace('/', ' ').replace('_', ' ').title()
                if key == 'validation/mean_average_precision':
                    ylabel = 'mAP'
                else: 
                    ylabel = title


                plt.locator_params(axis='x', nbins=10)
                plt.grid(True)
                plt.title(title)
                plt.xlabel('Global Steps')
                plt.ylabel(ylabel)
            plt.legend(loc='lower right')
            # Save the plot as a PDF file with the title as the filename
            filename = f"{key.replace('/', '_')}.pdf"
            plt.savefig(os.path.dirname(saveing_destination) + '/' + filename, format='pdf')
            # Clear the plot for the next iteration
            plt.clf()
            #if key == 'validation/loss':
            #    y_values = values
            #    #for i in range(1, len(values)):
            #    #    y_values[i] = 0.6 * values[i] + 0.4 * values[i-1] 
            #    
            #    y_values = np.repeat(y_values, len(x_values) // len(x_original))
            #    y_values = adjust_vector_length(x_values, y_values)
            #    #y_values = np.gradient(y_values)
            #    plt.plot(x_values, y_values, label='validation loss', linewidth=1.2, color='0.3')
            #    # Customize plot appearance
            #    plt.grid(True)
            #    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
            #    title = key.replace('/', ' ').replace('_', ' ').title()
            #    if key == 'validation/mean_average_precision':
            #        ylabel = 'mAP'
            #    else: 
            #        ylabel = title
            #    plt.locator_params(axis='x', nbins=16)
            #    plt.grid(True)
            #    plt.title('Validation Loss')
            #    plt.xlabel('Global Steps')
            #    plt.ylabel(ylabel)
            #    plt.legend(loc='upper right')
            #    #plt.ylim(-0.4, 0.1)
            #                # Save the plot as a PDF file with the title as the filename
            #    #filename = "LDAS.pdf"
            #    #filename = "smoothed_loss.pdf"
            #    filename = "valid_loss.pdf"
            #    plt.savefig(os.path.dirname(saveing_destination) + '/' + filename, format='pdf')
            #    # Clear the plot for the next iteration
            #    plt.clf()

# For usage fill in lists:
file_paths = []

#resets = reset_steps('/mnt/qb/work/hennig/hmx125/MLCommonsWorkspace/experiment_runs/multiple_cosine_ogbg_full.alt_warmups_no_opt.02/ogbg_jax/trial_1/resets.csv')
resets = []
data_dict = read_csv_to_dict(file_paths)
list_of_labels = []
list_of_colors = []
plot_and_save_as_pdf(data_dict, measures_of_interest, file_paths[0], list_of_labels, list_of_colors, resets)


