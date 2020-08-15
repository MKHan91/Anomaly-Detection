import os
import sys
import shutil


BASE_PATH = '/home/onepredict/Myungkyu/BVMS_turbine'

def check_dir_or_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def code_save_server(args, calendar):
    """dataloader, main, model code save"""
    dir = os.path.dirname(os.getcwd())

    main_code = sys.argv[0].split('/')[-1]
    x = main_code.split('_')

    x.remove(x[1])
    x.insert(1, 'dataloader')
    dataloader_code = '_'.join(x)

    x.remove(x[1])
    x.insert(1, 'model')
    x.remove(x[0])
    x.insert(0, 'generate')
    model_code = '_'.join(x)

    SAVE_CODE_PATH = os.path.join(BASE_PATH, '02_About_result', '01_Training_Result', '02_AutoEncoder_code', args.model_name + '_' + '_'.join(calendar[0:3]))
    check_dir_or_create(dir=SAVE_CODE_PATH)
    check_dir_or_create(dir=os.path.join(SAVE_CODE_PATH, args.model_name + '_' + '_'.join(calendar)))

    shutil.copy(os.path.join(dir, 'Server', dataloader_code),
                os.path.join(SAVE_CODE_PATH, args.model_name + '_' + '_'.join(calendar)))

    shutil.copy(os.path.join(dir, 'Server', main_code),
                os.path.join(SAVE_CODE_PATH, args.model_name + '_' + '_'.join(calendar)))

    shutil.copy(os.path.join(dir, 'Server', model_code),
                os.path.join(SAVE_CODE_PATH, args.model_name + '_' + '_'.join(calendar)))
