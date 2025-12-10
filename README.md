# CT-UAV-R-D---Nguyen Duc Quoc Binh Technical-Test
Technical Test to apply for R&amp;D Department about the solution for few-shot object detection.
The requirement.txt for installing library for project
There are code for demo in this repo including: prompt.yaml, prototype.py, demo.py
Pictures including: Support_image for create feature prototype, Test_image for testing and demo, the result after run demo
how to run demo:
pip install -r requirment.txt
python prototype.py --support_dir support_images --prompts prompts.yaml --out prototypes.pth 
python demo.py --prototype prototypes.pth --source test_image --out_path output --prompts_path prompts.yaml
