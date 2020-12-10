import expression_recognition
from train_options import ToolOptions

if __name__ == '__main__':
    opt = ToolOptions().parse()

    model = expression_recognition.ExpRecognition()
    model.prepare_devices(opt.gpu_ids, opt.landmark_num, opt.image_width)
    
    model.load_test_data(opt.test_image_path, opt.num_workers)
    model.prepare_tool(model_path = opt.model_path)
    model.test(opt.test_path, 0)