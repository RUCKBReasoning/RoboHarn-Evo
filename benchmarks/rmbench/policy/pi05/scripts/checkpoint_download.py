from pathlib import Path                                                                                                                                     
from huggingface_hub import snapshot_download                                                                                                                
                                                                                                                                                            
REPO_ID = "motus-robotics/pi0.5_robotwin2"                                                                                                                   
CHECKPOINT_NAME = "pi0.5_robotwin2"                                                                                                                          
                                                                                                                                                            
repo_root = Path(__file__).resolve().parents[3]
target_dir = repo_root / "policy" / "pi05" / "checkpoints" / CHECKPOINT_NAME
                                                                                                                                                            
target_dir.parent.mkdir(parents=True, exist_ok=True)
                                                                                                                                                            
local_path = snapshot_download(                                                                                                                              
    repo_id=REPO_ID,                                                                                                                                         
    repo_type="model",                                                                                                                                       
    local_dir=str(target_dir),                                                                                                                               
    local_dir_use_symlinks=False,                                                                                                                            
    resume_download=True,                                                                                                                                    
    endpoint="https://hf-mirror.com",                                                                                                                        
    max_workers=4,                                                                                                                                           
)                                                                                                                                                            
                                                                                                                                                            
print(f"Downloaded repo: {REPO_ID}")                                                                                                                         
print(f"Local path: {local_path}")                                                                                                                           
print(f"Checkpoint directory: {target_dir}") 
