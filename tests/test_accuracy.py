import torch
import time
import wandb
import os
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- IMPORT CORRETTI CON PREFISSO SRC ---

# 1. Importa i modelli SHARP
from src.wifi_doppler.models.sharp import (
    MultiAntennaClassifier, 
    SharpLegacySingleAntennaClassifier
)

# 2. Importa la factory per costruire il tuo generatore 
from src.wifi_doppler.models.csi_to_doppler_builders import build_csi_to_doppler_model

# 3. Importa il Dataset
from src.wifi_doppler.data.csi_to_sharp_doppler_dataset import CsiToSharpDopplerDataset

# ----------------------------------------

def evaluate_models(
    generator_dir, 
    sharp_weights_path, 
    dataloader, 
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Esecuzione su dispositivo: {device}")
    
    # 1. Inizializzazione del Modello Giudice (SHARP)
    print("Caricamento del classificatore SHARP...")
    # Dai tuoi file deduco che MultiAntennaClassifier avvolge un modello per singola antenna
    single_antenna_model = SharpLegacySingleAntennaClassifier(num_classes=4) # Adatta num_classes se diverso
    sharp_model = MultiAntennaClassifier(single_antenna_model=single_antenna_model, embedding_fusion="mean")
    
    sharp_model.load_state_dict(torch.load(sharp_weights_path, map_location=device))
    sharp_model.to(device)
    sharp_model.eval() # Congelato: garantisce nessun calcolo dei gradienti e disattiva dropout/batchnorm
    
    # Trova tutti i modelli generativi da valutare
    generator_models = glob.glob(os.path.join(generator_dir, "*.pt"))
    print(f"Trovati {len(generator_models)} modelli da valutare.")

    for gen_weights_path in generator_models:
        model_name = os.path.basename(gen_weights_path)
        
        # Inizializza un nuovo run su W&B
        wandb.init(
            project="csi-doppler-evaluation",
            name=f"eval_{model_name}",
            config={"model_name": model_name}
        )
        
        # 2. Inizializzazione del Generatore Corrente
        print(f"\nValutazione del modello: {model_name}")
        
        # Uso la tua factory. Assicurati che l'architettura scelta qui sia 
        # quella effettivamente usata per addestrare i .pt che stai testando.
        generator_config = {
            "architecture": "unet2d_shared_antenna_full_resolution", # o l'architettura che hai usato
            "num_antennas": 4,
            "num_subcarriers": 242,  # Regola in base a come hai addestrato
            "input_parts": 2,
            "output_doppler_bins": 100,
            "output_time": 340
        }
        generator = build_csi_to_doppler_model(generator_config)
        generator.load_state_dict(torch.load(gen_weights_path, map_location=device))
        generator.to(device)
        generator.eval()

        total_samples = 0
        agreed_predictions = 0        # Fidelity (Accordo tra SHARP_real e SHARP_gen)
        total_inference_time = 0.0    

        # 3. Ciclo di Inferenza (Senza Gradiente)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inferenza"):
                # Adattato al tuo dataset che restituisce: input, target, filename
                inputs, real_doppler, filenames = batch
                
                inputs = inputs.to(device)
                real_doppler = real_doppler.to(device)
                batch_size = inputs.size(0)
                total_samples += batch_size

                # Misurazione sicura del tempo su CUDA
                if device == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    
                    gen_doppler = generator(inputs)
                    
                    end_event.record()
                    torch.cuda.synchronize()
                    total_inference_time += start_event.elapsed_time(end_event) / 1000.0 
                else:
                    start_time = time.perf_counter()
                    gen_doppler = generator(inputs)
                    total_inference_time += (time.perf_counter() - start_time)

                # 4. Inoltro al Giudice (SHARP)
                # Calcola le previsioni sul doppler reale e su quello generato
                # Nota: MultiAntennaClassifier supporta argomenti di fusion, default è 'sum' 
                # (lo chiamo in base alla tua implementazione in sharp.py)
                real_logits = sharp_model(real_doppler, fusion="mean") 
                real_preds = real_logits.argmax(dim=-1)

                gen_logits = sharp_model(gen_doppler, fusion="mean")
                gen_preds = gen_logits.argmax(dim=-1)

                # 5. Calcolo Metriche
                # Quante volte SHARP predice esattamente la stessa classe (Fidelity)
                agreed_predictions += (gen_preds == real_preds).sum().item()

        # Calcoli finali aggregati
        fidelity_percent = (agreed_predictions / total_samples) * 100.0
        avg_time_per_trace_ms = (total_inference_time / total_samples) * 1000.0 

        print(f"Risultati {model_name}:")
        print(f" - Fidelity (Agreement tra originale e generato): {fidelity_percent:.2f}% ({agreed_predictions}/{total_samples})")
        print(f" - Tempo Inferenza Medio: {avg_time_per_trace_ms:.2f} ms")

        # 6. Logging su W&B
        wandb.log({
            "eval/fidelity_percent": fidelity_percent,
            "eval/agreed_instances": agreed_predictions,
            "eval/total_instances": total_samples,
            "eval/avg_inference_time_ms": avg_time_per_trace_ms
        })
        
        wandb.finish()


if __name__ == "__main__":
    GENERATOR_DIR = "E:/Repos/wifi-doppler-har/experiments/runs/trained_models"
    SHARP_WEIGHTS = "E:/Repos/wifi-doppler-har/experiments/runs/sharp_baseline/checkpoint_sharp.pt"
    
    print("Inizializzazione Dataset e DataLoader...")
    # Usa le tue reali cartelle
    dataset = CsiToSharpDopplerDataset(
        raw_root="data/CSI-80Mhz",
        doppler_root="data/doppler_traces",
        scenarios=("S1a", "S1b", "S1c"), # Assicurati di mettere gli scenari di Test/Val
        split=(0.6, 1.0)
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    evaluate_models(GENERATOR_DIR, SHARP_WEIGHTS, dataloader)