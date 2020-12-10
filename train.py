import expression_recognition
from train_options import ToolOptions

if __name__ == '__main__':
    opt = ToolOptions().parse()

    model = expression_recognition.ExpRecognition()
    model.prepare_devices(opt.gpu_ids, opt.landmark_num, opt.image_width)
    
    if opt.mode == 'train':
        model.load_train_data(opt.train_image_path, opt.train_label_path, opt.batch_size, opt.num_workers)
        model.load_validation_data(opt.validation_image_path, opt.validation_label_path, opt.num_workers)
        model.load_test_data(opt.test_image_path, opt.num_workers)
        model.prepare_tool(opt.start_lr, opt.learning_rate_decay_start, opt.total_epoch, opt.model_path, \
            opt.beta, opt.margin_1, opt.margin_2, opt.relabel_epoch)

        model.validation(opt.validation_path, 0)
        model.test(opt.test_path, 0)
        for epoch in range(1, opt.total_epoch+1):
            model.train(epoch)
            if epoch % opt.validation_frequency == 0:
                model.validation(opt.validation_path, epoch)
            if epoch % opt.save_frequency == 0:
                model.save_model(epoch, opt.save_path)
        model.test(opt.test_path, epoch)