import os

try:
    import torch
    import pandas as pd
    import numpy as np
    from transformers import BertTokenizer, BertModel, BertConfig
    from scipy.stats import spearmanr
    from utils_ZuCo import DataTransformer
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Missing Libraries, please install requirements.txt using")
    print(">> pip install -r requirements.txt")
    exit()

def check_filestructure():
    # Check if the current folder contains the "results_zuco" directory
    if not os.path.isdir("results_zuco"):
        raise FileNotFoundError("The folder 'results_zuco' is missing in the current directory. Please ensure the script is in the correct folder.")

    # Check if "results_zuco" contains a subfolder "task2"
    task2_path = os.path.join("results_zuco", "task2")
    if not os.path.isdir(task2_path):
        # Look for similarly named folders
        subfolders = [f for f in os.listdir("results_zuco") if os.path.isdir(os.path.join("results_zuco", f)) and "task2" in f]
        if subfolders:
            raise FileNotFoundError(f"The folder 'task2' is missing or incorrectly named. Did you mean: {subfolders[0]}? Please rename the folder to 'task2'.")
        else:
            raise FileNotFoundError("The folder 'task2' is missing in 'results_zuco'. Please ensure the file structure is correct.")

    # Check if "task2" contains 12 .mat files
    mat_files = [f for f in os.listdir(task2_path) if f.endswith(".mat")]
    if len(mat_files) != 12:
        raise FileNotFoundError(f"The folder 'task2' should contain 12 .mat files, but {len(mat_files)} were found. Please check the file structure.")
    all_files = [f for f in os.listdir(task2_path)]
    if len(all_files) != len(mat_files):
        print("The directory contains more than just the 12 required .mat files. This may cause utils_ZuCo to throw an error. Please remove the extraneous files.")
        raise Exception

def main():
    # Check the file structure
    check_filestructure()

    # Ask the user for input paths
    sentence_file = input("Enter the path to the sentence file (e.g., results_zuco/task_materials/relations_labels_task2.csv): ").strip()
    output_file = input("Enter the path to save the output CSV file (e.g., correlation_results.csv): ").strip()

    # Load ZuCo Eye-Tracking data
    data_transformer = DataTransformer("task2", level='word', scaling='raw', fillna='zeros')
    print("Succesfully loaded the data transformer, loading ZuCo Data (Expect some errors to appear)")

    participant_dataframes = {}
    for i in range(12):
        try:
            df = data_transformer(i)
            participant_dataframes[i] = df
        except Exception as e:
            print(f"Error processing subject {i}: {e}")

    print(f"Number of participants succesfully loaded: {len(participant_dataframes)}")
    print(f"Finished loading ZuCo Data")

    print("Loading BERT models")

    # Initialize BERT models and tokenizers
    monolingual_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
    multilingual_config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True, output_attentions=True)

    monolingual_model = BertModel.from_pretrained('bert-base-uncased', config=monolingual_config)
    multilingual_model = BertModel.from_pretrained('bert-base-multilingual-cased', config=multilingual_config)

    monolingual_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    multilingual_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    monolingual_model.eval()
    multilingual_model.eval()

    print("Finished loading BERT models")

    # Load sentence data
    sentence_data = pd.read_csv(sentence_file, header=None, names=['ID', 'Part', 'Sentence', 'Empty', 'Relation'])
    sentences = sentence_data['Sentence']

    def align_tokens_to_words(words, tokens):
        aligned = []
        token_idx = 1
        for word in words:
            sub_tokens = []
            reconstructed_word = ""
            while token_idx < len(tokens) - 1:
                token = tokens[token_idx]
                if token.startswith("##"):
                    token = token[2:]
                reconstructed_word += token
                if reconstructed_word.lower() == word.lower():
                    sub_tokens.append(token_idx)
                    token_idx += 1
                    break
                elif word.lower().startswith(reconstructed_word.lower()):
                    sub_tokens.append(token_idx)
                    token_idx += 1
                else:
                    break
            aligned.append(sub_tokens if reconstructed_word.lower() == word.lower() else [])
        return aligned

    def process_zuco_participant_data(participant_dataframes, monolingual_model, monolingual_tokenizer,
                                      multilingual_model, multilingual_tokenizer):
        results = []
        for participant_id, participant_data in participant_dataframes.items():
            print(f"Processing data for participant {participant_id}...")

            participant_data.columns = [col[0] for col in participant_data.columns]

            for sent_id, group in participant_data.groupby("Sent_ID"):
                sent_id = int(sent_id.rstrip("_NR"))
                words = group["Word"].tolist()
                human_attention = group["TRT"].tolist()
                sentence = " ".join(words)

                mono_inputs = monolingual_tokenizer(sentence, return_tensors="pt", truncation=True)
                multilingual_inputs = multilingual_tokenizer(sentence, return_tensors="pt", truncation=True)

                with torch.no_grad():
                    mono_outputs = monolingual_model(**mono_inputs)
                    multilingual_outputs = multilingual_model(**multilingual_inputs)

                mono_first_layer_cls = mono_outputs.attentions[0][:, :, 0, :]
                mono_last_layer_cls = mono_outputs.attentions[-1][:, :, 0, :]
                multilingual_first_layer_cls = multilingual_outputs.attentions[0][:, :, 0, :]
                multilingual_last_layer_cls = multilingual_outputs.attentions[-1][:, :, 0, :]

                mono_first_layer_attention = mono_first_layer_cls.mean(dim=1).squeeze(0)
                mono_last_layer_attention = mono_last_layer_cls.mean(dim=1).squeeze(0)
                multilingual_first_layer_attention = multilingual_first_layer_cls.mean(dim=1).squeeze(0)
                multilingual_last_layer_attention = multilingual_last_layer_cls.mean(dim=1).squeeze(0)

                mono_aligned = align_tokens_to_words(
                    words, monolingual_tokenizer.convert_ids_to_tokens(mono_inputs["input_ids"].squeeze())
                )
                multilingual_aligned = align_tokens_to_words(
                    words, multilingual_tokenizer.convert_ids_to_tokens(multilingual_inputs["input_ids"].squeeze())
                )

                mono_first_word_attention = [
                    mono_first_layer_attention[idx].mean().item() if len(idx) > 0 else 0
                    for idx in mono_aligned
                ]
                mono_last_word_attention = [
                    mono_last_layer_attention[idx].mean().item() if len(idx) > 0 else 0
                    for idx in mono_aligned
                ]
                multilingual_first_word_attention = [
                    multilingual_first_layer_attention[idx].mean().item() if len(idx) > 0 else 0
                    for idx in multilingual_aligned
                ]
                multilingual_last_word_attention = [
                    multilingual_last_layer_attention[idx].mean().item() if len(idx) > 0 else 0
                    for idx in multilingual_aligned
                ]

                def safe_spearmanr(arr1, arr2):
                    if len(set(arr1)) > 1 and len(set(arr2)) > 1:
                        return spearmanr(arr1, arr2).correlation
                    else:
                        return float("nan")

                mono_first_corr = safe_spearmanr(human_attention, mono_first_word_attention)
                mono_last_corr = safe_spearmanr(human_attention, mono_last_word_attention)
                multilingual_first_corr = safe_spearmanr(human_attention, multilingual_first_word_attention)
                multilingual_last_corr = safe_spearmanr(human_attention, multilingual_last_word_attention)

                results.append({
                    "Participant_ID": participant_id,
                    "Sentence_ID": sent_id,
                    "Mono_First_Corr": mono_first_corr,
                    "Mono_Last_Corr": mono_last_corr,
                    "Multilingual_First_Corr": multilingual_first_corr,
                    "Multilingual_Last_Corr": multilingual_last_corr,
                })

        return pd.DataFrame(results)

    results_df = process_zuco_participant_data(
        participant_dataframes, monolingual_model, monolingual_tokenizer,
        multilingual_model, multilingual_tokenizer
    )

    results_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")

    # Generate summary statistics and plots
    def plot_results(correlation_results):
        plt.figure(figsize=(12, 6))
        participants = correlation_results.groupby("Participant_ID").mean().reset_index()
        participants["Participant_ID"] = range(1, len(participants) + 1)
        sns.lineplot(x="Participant_ID", y="Mono_First_Corr", data=participants, label="Mono First Layer", marker="o")
        sns.lineplot(x="Participant_ID", y="Mono_Last_Corr", data=participants, label="Mono Last Layer", marker="o")
        sns.lineplot(x="Participant_ID", y="Multilingual_First_Corr", data=participants, label="Bi First Layer", marker="o")
        sns.lineplot(x="Participant_ID", y="Multilingual_Last_Corr", data=participants, label="Bi Last Layer", marker="o")
        plt.title("Correlations Per Participant Across Layers and Models")
        plt.xlabel("Participant ID")
        plt.ylabel("Average Correlation")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("plot_results.png")

    correlation_results = pd.read_csv(output_file)
    plot_results(correlation_results)

if __name__ == "__main__":
    main()
