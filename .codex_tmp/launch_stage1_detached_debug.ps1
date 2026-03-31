$env:PYTHONUNBUFFERED = '1'
Set-Location 'D:\project-v&t'
& 'C:\Users\shaw1\AppData\Local\Programs\Python\Python310\python.exe' scripts/train.py --project-root . --stage configs/stages/stage1_perception_fold1.yaml
