from pathlib import Path

def create_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def main():
    ROOT_PATH = "/home/jovyan/ChestXray-14"
    create_path(f"{ROOT_PATH}/results/models/facal_loss/EfficientNetB0/")

if __name__ == "__main__":
    main()