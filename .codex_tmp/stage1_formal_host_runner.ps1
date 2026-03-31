$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
Set-Location 'D:\project-v&t'
& 'C:\Users\shaw1\AppData\Local\Programs\Python\Python310\python.exe' scripts/train.py --project-root . --stage 'configs/stages/stage1_perception_fold1.yaml' --resume 'D:\project-v&t\runs\stage1_perception_fold1\checkpoints\stage1_latest.pt' 1>>'D:\project-v&t\runs\stage1_perception_fold1\hosted_stage1.stdout.log' 2>>'D:\project-v&t\runs\stage1_perception_fold1\hosted_stage1.stderr.log'
