import json
import os

import SimpleITK
import torch

from batchgenerators.utilities.file_and_folder_operations import join, isdir

from predict_cls import generate_mip, trace_classification
from predict_seg import predict_from_folder

class Autopet_baseline:

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        # self.nii_seg_file = "TCIA_001.nii.gz"
        self.output_path_category = "/output/data-centric-model.json"
        self.classfication_json = "/output/output_cls.json"

        self.mip_path   = "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/mipTs"
        self.model_path = '/opt/algorithm/models'
        self.trace_cls_modelfile  = os.path.join(self.model_path, 'cls_model', 'best.pt')
        self.training_output_path = os.path.join(self.model_path, 'seg_model')


    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mhas = os.listdir(os.path.join(self.input_path, "images/ct/"))
        pet_mhas = os.listdir(os.path.join(self.input_path, "images/pet/"))
        uuids = []
        for ct_mha in ct_mhas:
            uuids.append(os.path.splitext(ct_mha)[0])
        # print(pet_mhas)

        subj = []
        for n, ct_mha in enumerate(ct_mhas):
            sub_name = f"TCIA_{str(n).zfill(3)}"
            # print(sub_name, ' Process')
            self.convert_mha_to_nii(
                os.path.join(self.input_path, "images/ct/", ct_mha),
                os.path.join(self.nii_path, f"{sub_name}_0000.nii.gz"),
            )
            self.convert_mha_to_nii(
                os.path.join(self.input_path, "images/pet/", pet_mhas[n]),
                os.path.join(self.nii_path, f"{sub_name}_0001.nii.gz"),
            )
            subj.append(sub_name)

        self.subj = subj
        self.uuid = uuids

        return uuids, subj

    def write_outputs(self, uuids, subjs):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        for u, uuid in enumerate(uuids):
            print(subjs[u])
            self.convert_nii_to_mha(
                os.path.join(self.result_path, subjs[u] + ".nii.gz"),
                os.path.join(self.output_path, uuid + ".mha"),
            )
            print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        """
        Your algorithm goes here
        """
        print("Trace classification starting!")
        generate_mip(self.nii_path, self.subj, self.mip_path, multithread=True, num_threads=10)
        fdg_subj, psma_subj = trace_classification(self.trace_cls_modelfile,
                                                   self.mip_path,
                                                   json_file=self.classfication_json)

        print("\n\nnnUNet segmentation starting!")
        input_folder  = self.nii_path
        output_folder = self.result_path

        save_npz                  = False  #args.save_npz
        lowres_segmentations      = 'None'  #args.lowres_segmentations
        num_threads_preprocessing = 1  #args.num_threads_preprocessing
        num_threads_nifti_save    = 1  # args.num_threads_nifti_save
        disable_tta               = True  #args.disable_tta
        step_size                 = 0.5
        overwrite_existing        = True  #args.overwrite_existing
        mode                      = 'normal'  #args.mode
        all_in_gpu                = 'None'  #args.all_in_gpu
        model                     = '3d_fullres'  # args.model
        disable_mixed_precision   = False  #args.disable_mixed_precision
        chk                       = 'model_final_checkpoint'

        task_name_psma = 'autopet+nsy+mi_427'
        folds_psma     = 1
        task_name_fdg  = 'autopet+lym+mi_902'
        folds_fdg      = 'all'

        task_name_fdg  = 'autopet_fdg_747'
        folds_fdg      = 'all'
        task_name_psma = 'autopet_psma_418'
        folds_psma     = 'all'

        assert all_in_gpu in ['None', 'False', 'True']
        if all_in_gpu == "None":
            all_in_gpu = None
        elif all_in_gpu == "True":
            all_in_gpu = True
        elif all_in_gpu == "False":
            all_in_gpu = False

        if lowres_segmentations == "None":
            lowres_segmentations = None

        segmentation_export_kwargs = {'force_separate_z'     : None,
                                      'interpolation_order'  : 1,
                                      'interpolation_order_z': 1,
                                      'npz_file'             : None}

        print('\n', '='*5, 'FDG SEGMENTATION')
        model_folder_name = join(self.training_output_path, task_name_fdg, model)
        print("using model stored in ", model_folder_name)
        assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
        predict_from_folder(model_folder_name, input_folder, output_folder, fdg_subj, folds_fdg, save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, not disable_tta,
                            lesion_id=[13],
                            overwrite_existing=overwrite_existing, 
                            mode=mode, 
                            overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size, 
                            checkpoint_name=chk,
                            segmentation_export_kwargs=segmentation_export_kwargs)
        
        print('\n', '='*5, 'PSMA SEGMENTATION')
        model_folder_name = join(self.training_output_path, task_name_psma, model)
        print("using model stored in ", model_folder_name)
        assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
        predict_from_folder(model_folder_name, input_folder, output_folder, psma_subj, folds_psma, save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, not disable_tta,
                            lesion_id=[13],
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size, checkpoint_name=chk,
                            segmentation_export_kwargs=segmentation_export_kwargs)
        
    
        # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but
        # segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being
        # printed.
        print("Prediction finished")

    def save_datacentric(self, value: bool):
        print("Saving datacentric json to " + self.output_path_category)
        with open(self.output_path_category, "w") as json_file:
            json.dump(value, json_file, indent=4)

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()

        print("="*10, "Start processing", "="*10)
        uuid, subj = self.load_inputs()

        print("="*10, "Start prediction", "="*10)
        self.predict()

        print("="*10, "Start output writing", "="*10)
        self.save_datacentric(False)
        self.write_outputs(uuid, subj)


if __name__ == "__main__":

    print("START")
    Autopet_baseline().process()
    # modality = 'ct'
    # nii_path = f'/home/jiayi/Projects/autoPET/code/preliminary_set/{modality}'
    # output_path = f'//home/jiayi/Projects/autoPET/code/preliminary_set/input/images/{modality}'
    # os.makedirs(output_path, exist_ok=True)

    # ab = Autopet_baseline()

    # for file in os.listdir(nii_path):
    #     filename = file.split('.nii.gz')[0]
    #     ab.convert_nii_to_mha(f'{nii_path}/{file}', f'{output_path}/{filename}.mha')