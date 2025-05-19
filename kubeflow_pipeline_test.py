from typing import Dict, List
from kfp import dsl
from kfp import compiler
import kfp
from client_manager import KFPClientManager
from kfp.dsl import Input, Output, Dataset, Model, component
# Preprocessing Function for Prediction 
def preprocess_input(raw_input: str) -> str:
    """
    Preprocess raw input (e.g., TrdCaptRpt string) to match input_message_clean format.
    Args:
        raw_input (str): Raw input string, e.g., concatenated TrdCaptRpt messages.
    Returns:
        str: Cleaned input string ready for model prediction.
    """
    import re
    # Remove XML-like tags and special characters
    pattern = r'[<"/]'
    cleaned_text = re.sub(pattern, '', raw_input)
    cleaned_text = cleaned_text.replace('>', ' ')
    # Remove extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

# Step 1: Generate Dataset
@component(base_image="tarakdhieb7/kubeflow-testing:latest")
def generate_data(output_csv: Output[Dataset], execution_id: str = ""):
    import subprocess
    subprocess.run(["pip", "install", "minio"], check=True)
    import random
    import pandas as pd
    import numpy as np
    from minio import Minio
    from minio.error import S3Error
    import os
    import uuid

    # Initialize minio client
    minio_client = Minio(
        "10.106.67.253:9000",
        access_key="minio",  # Replace with your minio access key
        secret_key="minio123",  # Replace with your minio secret key
        secure=False
    )

    # Ensure bucket exists
    bucket_name = "mlpipeline"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    def generate_full_service_wf():
        rptid = random.randint(1000, 400000)
        trdid = random.randint(20281004800, 40281004800)
        trdtyp = random.choice([0, 1, 2, 3, 11, 15, 20, 29, 31, 45])
        mtchid = random.randint(20281001777, 40281001777)
        pxtyp = random.choice([2, 10])
        lastqty = random.randint(10, 100)
        lastpx = random.choice([100, 370, 876, 250.45, 300.9, 540.698, 1050, 3590.59])
        trdt = random.choice(["2022-12-17", "2022-12-21", "2023-01-02", "2023-01-05"])
        bizdt = random.choice(["2023-01-06", "2023-01-11", "2023-01-20", "2023-02-05", "2023-02-10"])
        t = random.choice(["2023-02-15", "2023-02-21", "2023-03-20", "2023-03-30"])
        time1 = random.choice(['T05:17:30', 'T10:18:15', 'T12:25:13', 'T16:40:55'])
        r1 = random.choice(['.100', '.230', '.326'])
        r2 = random.choice(['.350', '.470', '.768'])
        tz1 = random.choice(['-5:00', '-6:00', '-7:00'])
        txntm = t + time1 + r1 + tz1
        time2 = random.choice(['T17:30:55.600', 'T19:33:05.333', 'T22:03:12.436'])
        snt1 = t + time1 + r2 + tz1
        snt2 = t + time2 + '+01:00'
        snt3 = t + 'T00:12:55.138' + tz1
        snt4 = t + 'T00:12:55.160' + tz1
        sym = random.choice(["BUI", "ABP", "TMP"])
        id5 = random.choice(['B', 'W', 'AS', 'LM', 'BUI', 'IP0'])
        my = random.choice(["2023-05", "2023-07", "2023-10"])
        mmy = my.replace('-', '')
        matdt = my + '-' + str(random.randint(1, 28))
        inptsrc = random.choice(['MA', 'EL', 'SY'])
        clordid = random.choice(['123', '134', '145', '146ZBT', '245BTN', '987NL', 'OrdAT'])
        tid = random.choice(['ABCD', 'D210', 'B509', 'HBPL', 'TL98'])
        id2 = random.choice(['Emira-115', 'Ahlem-110', 'Karim-200', 'Mohamed-300', 'acc-AT', 'bl-NL', 'Mariem', 'ahmed'])
        typ = random.randint(20, 30)
        id3 = random.choice(["FIRMACT1", "AKLPM9", "MPLAMP6", "00120007", "Hassen", "Ahlem", 'arij'])
        custcpcty1 = random.randint(1, 4)
        custcpcty2 = random.randint(1, 4)
        ex = random.choice(['BTNL', 'SNTL', 'HDPL', 'XMGE'])
        mult = random.choice([0.1, 0.2, 0.3, 0.4, 0.9, 1000, 2000, 5000, 300, 200, 50, 70])
        pos1 = random.choice(['Y', 'N'])
        pos3 = random.choice(['Y', 'N'])
        cfi = random.choice(['FCCPXX', 'APYBKL', 'MKLPON', 'FCAPSX'])
        pos2 = random.choice(['O', 'C'])
        side = random.randint(1, 2)
        id26 = random.randint(1, 2)
        mlrty = random.randint(1, 3)
        alin = random.randint(0, 1)
        inpdev = random.choice(['PORTAL', 'EXCHANGE', 'API', 'CLEARING'])
        id4 = random.randint(100, 500)
        id12 = random.choice(['100', '200', '400', 'A123', 'B115'])
        cfi2 = random.choice(['FCCPXX', 'APYBKL', 'MKLPON', 'FCAPSX'])
        return (
            f"""<TrdCaptRpt RptID="{rptid}" TrdID="{trdid}" TransTyp="0" RptTyp="2" TrdTyp="{trdtyp}" MtchID="{mtchid}" PxTyp="{pxtyp}" LastQty="{lastqty}" LastPx="{lastpx}" TrdDt="{trdt}" BizDt="{bizdt}" TxnTm="{txntm}"><Hdr SID="MGEX" TID="{tid}" PosDup="{pos1}" PosRsnd="{pos3}" Snt="{snt1}"></Hdr>
        <Instrmt Sym="{sym}" ID="{id5}" Src="H" CFI="{cfi}" MMY="{mmy}" MatDt="{matdt}" Mult="{mult}" Exch="{ex}"></Instrmt><RptSide Side="{side}" Ccy="USD" InptSrc="{inptsrc}" InptDev="{inpdev}" CustCpcty="{custcpcty1}" PosEfct="{pos2}" ClOrdID="{clordid}" MLegRptTyp="{mlrty}" AllocInd="{alin}"><Pty ID="MGEX" R="21"></Pty><Pty ID="210" R="1"></Pty><Pty ID="{id4}" R="4"></Pty>
        <Pty ID="{id2}" R="24"><Sub ID="{id26}" Typ="26"></Sub></Pty><Pty ID="{id12}" R="12"></Pty><TrdRegTS TS="{txntm}" Typ="1">
        </TrdRegTS></RptSide></TrdCaptRpt>""".replace('\n', ''),
            f"""<TrdCaptRpt RptID="{trdid}" TrdDt="{trdt}" BizDt="{bizdt}" TxnTm="{txntm}" TrdID="{trdid}" TransTyp="2" RptTyp="0" LastQty="{float(lastqty)}" LastPx="{float(lastpx)}"><Hdr Snt="{snt2}" TID="MGEX" SID="{tid}"/><Instrmt Exch="{ex}" ID="{id5}" MMY="{mmy}" CFI="{cfi2}"/><RptSide Side="{side}" CustCpcty="{custcpcty2}">
        <Pty ID="210" R="1"/><Pty ID="{id3}" R="24"><Sub ID="{id26}" Typ="26"/></Pty></RptSide></TrdCaptRpt>""".replace('\n', ''),
            f"""<TrdCaptRpt RptID="580198" TrdID="{trdid}" TransTyp="2" RptTyp="2" TrdTyp="{trdtyp}" MtchID="{mtchid}" PxTyp="{pxtyp}" LastQty="{lastqty}" LastPx="{lastpx}" TrdDt="{trdt}" BizDt="{bizdt}" TxnTm="{snt3}"><Hdr SID="MGEX" TID="{tid}" PosDup="{pos1}" PosRsnd="{pos3}" Snt="{snt4}"></Hdr>
        <Instrmt Sym="{sym}" ID="{id5}" Src="H" CFI="{cfi}" MMY="{mmy}" MatDt="{matdt}" Mult="{mult}" Exch="{ex}"></Instrmt><RptSide Side="{side}" Ccy="USD" InptSrc="{inptsrc}" InptDev="{inpdev}" CustCpcty="{custcpcty2}" PosEfct="{pos2}" ClOrdID="{clordid}" MLegRptTyp="{mlrty}" AllocInd="{alin}"><Pty ID="MGEX" R="21"></Pty><Pty ID="210" R="1"></Pty><Pty ID="{id4}" R="4"></Pty>
        <Pty ID="{id3}" R="24"><Sub ID="{id26}" Typ="26"></Sub></Pty><Pty ID="{id12}" R="12"></Pty><TrdRegTS TS="{txntm}" Typ="1"></TrdRegTS></RptSide>
        </TrdCaptRpt>""".replace('\n', '')
        )

    dataset = [generate_full_service_wf() for _ in range(500)]
    ccp_new_trade_messages = [message[0] for message in dataset]
    sgw_fs_operation_messages = [message[1] for message in dataset]
    ccp_response_messages = [message[2] for message in dataset]
    df = pd.DataFrame({
        'Executed Trade': ccp_new_trade_messages,
        'Full Service Operation': sgw_fs_operation_messages,
        'CCP Response': ccp_response_messages
    })
    df['input_message'] = df['Executed Trade'] + ' ' + df['Full Service Operation']
    df.drop(['Executed Trade', 'Full Service Operation'], axis=1, inplace=True)
    df.to_csv(output_csv.path, index=False)
    print(output_csv.path)
    # Upload to minio
    execution_id = execution_id if execution_id else str(uuid.uuid4())
    object_name = f"{execution_id}/generate_data/output.csv"
    try:
        minio_client.fput_object(bucket_name, object_name, output_csv.path)
        print(f"Uploaded {object_name} to minio bucket {bucket_name}")
    except S3Error as e:
        print(f"Failed to upload to minio: {e}")

# Step 2: Preprocess Data
@component(base_image="tarakdhieb7/kubeflow-testing:latest")
def preprocess_data(input_csv: Input[Dataset], output_train: Output[Dataset], output_val: Output[Dataset], execution_id: str = ""):
    import subprocess
    subprocess.run(["pip", "install", "minio"], check=True)
    import re
    import pandas as pd
    from minio import Minio
    from minio.error import S3Error
    import uuid

    # Initialize minio client
    minio_client = Minio(
        "10.106.67.253:9000",
        access_key="minio",  # Replace with your minio access key
        secret_key="minio123",  # Replace with your minio secret key
        secure=False
    )

    # Ensure bucket exists
    bucket_name = "mlpipeline"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    df = pd.read_csv(input_csv.path)
    def clean_text(input_message):
        pattern = r'[<"/]'
        cleaned_text = re.sub(pattern, '', input_message)
        cleaned_text = cleaned_text.replace('>', ' ')
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text

    df['input_message_clean'] = df['input_message'].apply(clean_text)
    df.drop(['input_message'], axis=1, inplace=True)
    train_data = df.iloc[:int(len(df) * 0.9)]
    val_data = df.iloc[int(len(df) * 0.9):]
    print("Shapes after train-val split:")
    print("train_df:", train_data.shape)
    print("val_df:", val_data.shape)
    train_data.to_csv(output_train.path, index=False)
    val_data.to_csv(output_val.path, index=False)
    print("Preprocessed data saved to:", output_train.path, output_val.path)

    # Upload to minio
    execution_id = execution_id if execution_id else str(uuid.uuid4())
    for output, name in [(output_train, "output_train.csv"), (output_val, "output_val.csv")]:
        object_name = f"{execution_id}/preprocess_data/{name}"
        try:
            minio_client.fput_object(bucket_name, object_name, output.path)
            print(f"Uploaded {object_name} to minio bucket {bucket_name}")
        except S3Error as e:
            print(f"Failed to upload to minio: {e}")

# Step 3: Train Model
@component(base_image="tarakdhieb7/kubeflow-testing:latest")
def train_model(train_data: Input[Dataset], val_data: Input[Dataset], model_output: Output[Model], execution_id: str = ""):
    import subprocess
    subprocess.run(["pip", "install", "minio"], check=True)

    import torch
    import pandas as pd
    from transformers import GPT2Tokenizer, AutoModelForCausalLM
    from torch.utils.data import Dataset, DataLoader
    import time
    import os
    from minio import Minio
    from minio.error import S3Error
    import uuid

    # Initialize minio client
    minio_client = Minio(
        "10.106.67.253:9000",
        access_key="minio",  # Replace with your minio access key
        secret_key="minio123",  # Replace with your minio secret key
        secure=False
    )

    # Ensure bucket exists
    bucket_name = "mlpipeline"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''

    class CustomDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            input_seq = self.data.iloc[index]['input_message_clean']
            target_seq = self.data.iloc[index]['CCP Response']
            concat_seq = f"{input_seq} {target_seq}"
            tokenized_seq = self.tokenizer.encode(
                concat_seq,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            return tokenized_seq.squeeze(0)

    # Load data
    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(val_data.path)
    print("Shape of train_df:", train_df.shape)
    print("Shape of val_df:", val_df.shape)

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', low_cpu_mem_usage=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')

    # Create datasets
    max_length = 512  # Reduced for CPU
    batch_size = 1
    train_dataset = CustomDataset(train_df, tokenizer, max_length)
    val_dataset = CustomDataset(val_df, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 1
    start_time = time.time()
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids = batch.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")
    print(f"Training Time: {(time.time() - start_time)/60:.2f} minutes")

    # Save model and tokenizer
    model.save_pretrained(model_output.path)
    tokenizer.save_pretrained(model_output.path)

    # Upload model files to minio
    execution_id = execution_id if execution_id else str(uuid.uuid4())
    model_files = [
        "config.json",
        "pytorch_model.bin",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    for file in model_files:
        file_path = os.path.join(model_output.path, file)
        if os.path.exists(file_path):
            object_name = f"{execution_id}/train_model/{file}"
            try:
                minio_client.fput_object(bucket_name, object_name, file_path)
                print(f"Uploaded {object_name} to minio bucket {bucket_name}")
            except S3Error as e:
                print(f"Failed to upload to minio: {e}")

@component(base_image="tarakdhieb7/kubeflow-testing:latest")
def train_model_distributed(train_data: Input[Dataset], val_data: Input[Dataset], model_output: Output[Model], execution_id: str = ""):
    import subprocess
    subprocess.run(["pip", "install", "minio"], check=True)
    import os
    import torch
    import torch.distributed as dist
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2Tokenizer, AutoModelForCausalLM
    from torch.nn.parallel import DistributedDataParallel as DDP
    import logging
    import time
    import sys
    from minio import Minio
    from minio.error import S3Error
    import uuid

    # Initialize minio client
    minio_client = Minio(
        "10.106.67.253:9000",
        access_key="minio",  # Replace with your minio access key
        secret_key="minio123",  # Replace with your minio secret key
        secure=False
    )

    # Ensure bucket exists
    bucket_name = "mlpipeline"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

# Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Starting train_model_distributed, execution_id: {execution_id}")
    logger.info(f"Model output path: {model_output.path}")

    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''

    class CustomDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            input_seq = self.data.iloc[index]['input_message_clean']
            target_seq = self.data.iloc[index]['CCP Response']
            concat_seq = f"{input_seq} {target_seq}"
            tokenized_seq = self.tokenizer.encode(
                concat_seq,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            return tokenized_seq.squeeze(0)

    def setup_ddp(rank, world_size):
        try:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12345'
            dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
            logger.info(f"DDP initialized for rank {rank}/{world_size}")
        except Exception as e:
            logger.error(f"DDP setup failed: {e}")
            raise

    def train(rank, world_size, train_loader, model, optimizer, device, epochs):
        try:
            setup_ddp(rank, world_size)
            model = model.to(device)
            model = DDP(model, device_ids=None)
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for i, batch in enumerate(train_loader):
                    input_ids = batch.to(device)
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    if rank == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                avg_loss = total_loss / len(train_loader)
                if rank == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            dist.destroy_process_group()
        except Exception as e:
            logger.error(f"Training failed on rank {rank}: {e}")
            dist.destroy_process_group()
            raise

    # Function to initialize the dataset and model (called in each process)
    def initialize_training_components():
        # Load data
        train_df = pd.read_csv(train_data.path)
        logger.info(f"Shape of train_df: {train_df.shape}")

        # Initialize tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', low_cpu_mem_usage=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained('distilgpt2')

        # Create dataset
        max_length = 512
        batch_size = 2
        train_dataset = CustomDataset(train_df, tokenizer, max_length)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler
        )

        # Training setup
        device = torch.device('cpu')
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        epochs = 1

        return train_loader, model, optimizer, device, epochs, tokenizer

    def distributed_training():
        try:
            # Get rank and world_size from environment (set by torchrun)
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            
            start_time = time.time()
            logger.info(f"Starting distributed training with rank={rank}, world_size={world_size}")

            # Initialize training components in each process
            train_loader, model, optimizer, device, epochs, tokenizer = initialize_training_components()

            # Run the training
            train(rank, world_size, train_loader, model, optimizer, device, epochs)

            # Save model and tokenizer (only rank 0)
            if rank == 0:
                model.module.save_pretrained(model_output.path)
                model.metadata['framework'] = 'transformers'
                tokenizer.save_pretrained(model_output.path)
                logger.info(f"Model saved to {model_output.path}")

                # Upload model files to minio
                execution_id = execution_id if execution_id else str(uuid.uuid4())
                model_files = [
                    "config.json",
                    "pytorch_model.bin",
                    "vocab.json",
                    "merges.txt",
                    "tokenizer_config.json",
                    "special_tokens_map.json"
                ]
                for file in model_files:
                    file_path = os.path.join(model_output.path, file)
                    if os.path.exists(file_path):
                        object_name = f"{execution_id}/train_model_distributed/{file}"
                        try:
                            minio_client.fput_object(bucket_name, object_name, file_path)
                            logger.info(f"Uploaded {object_name} to minio bucket {bucket_name}")
                        except S3Error as e:
                            logger.error(f"Failed to upload to minio: {e}")
            logger.info(f"Training Time: {(time.time() - start_time)/60:.2f} minutes")
        except Exception as e:
            logger.error(f"Distributed training failed: {e}")
            raise

    if __name__ == "__main__":
        # Since this is a Kubeflow component, we'll assume it's invoked with torchrun
        # Example command (not run here, but for reference):
        # torchrun --nproc_per_node=3 --master_addr=localhost --master_port=12345 this_script.py
        distributed_training()

# Step 4: Evaluate Model
@component(base_image="tarakdhieb7/kubeflow-testing:latest")
def evaluate_model(val_data: Input[Dataset], model: Input[Model], metrics_output: Output[Dataset], execution_id: str = ""):
    import subprocess
    subprocess.run(["pip", "install", "minio"], check=True)
    import torch
    import pandas as pd
    from transformers import GPT2Tokenizer, AutoModelForCausalLM
    from torch.utils.data import Dataset, DataLoader
    import time
    from minio import Minio
    from minio.error import S3Error
    import uuid

    # Initialize minio client
    minio_client = Minio(
        "10.106.67.253:9000",
        access_key="minio",  # Replace with your minio access key
        secret_key="minio123",  # Replace with your minio secret key
        secure=False
    )

    # Ensure bucket exists
    bucket_name = "mlpipeline"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    class CustomDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            input_seq = self.data.iloc[index]['input_message_clean']
            target_seq = self.data.iloc[index]['CCP Response']
            concat_seq = f"{input_seq} {target_seq}"
            tokenized_seq = self.tokenizer.encode(
                concat_seq,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            return tokenized_seq.squeeze(0)

    # Load data
    val_df = pd.read_csv(val_data.path)
    print("Shape of val_df:", val_df.shape)

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model.path)
    model = AutoModelForCausalLM.from_pretrained(model.path)
    max_length = 512
    val_dataset = CustomDataset(val_df, tokenizer, max_length)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    total_loss = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Average Validation Loss: {avg_loss:.4f}")
    print(f"Validation Time: {(time.time() - start_time)/60:.2f} minutes")

    # Save metrics
    with open(metrics_output.path, 'w') as f:
        f.write(f"Average Validation Loss: {avg_loss:.4f}")

    # Upload to minio
    execution_id = execution_id if execution_id else str(uuid.uuid4())
    object_name = f"{execution_id}/evaluate_model/metrics.txt"
    try:
        minio_client.fput_object(bucket_name, object_name, metrics_output.path)
        print(f"Uploaded {object_name} to minio bucket {bucket_name}")
    except S3Error as e:
        print(f"Failed to upload to minio: {e}")

# Step 5: Predict
@component(base_image="tarakdhieb7/kubeflow-testing:latest")
def predict(val_data: Input[Dataset], model: Input[Model], prediction_output: Output[Dataset], execution_id: str = ""):
    import subprocess
    subprocess.run(["pip", "install", "minio"], check=True)
    import torch
    import pandas as pd
    from transformers import GPT2Tokenizer, AutoModelForCausalLM
    import time
    from minio import Minio
    from minio.error import S3Error
    import uuid

    # Initialize minio client
    minio_client = Minio(
        "10.106.67.253:9000",
        access_key="minio",  # Replace with your minio access key
        secret_key="minio123",  # Replace with your minio secret key
        secure=False
    )

    # Ensure bucket exists
    bucket_name = "mlpipeline"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    def preprocess_input(raw_input: str) -> str:
        import re
        pattern = r'[<"/]'
        cleaned_text = re.sub(pattern, '', raw_input)
        cleaned_text = cleaned_text.replace('>', ' ')
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text

    def predict_single(input_message: str, model, tokenizer, device, max_length=512):
        model.eval()
        cleaned_input = preprocess_input(input_message)
        tokenized_input = tokenizer.encode(
            cleaned_input,
            return_tensors='pt',
            max_length=max_length,
            truncation=True
        )
        tokenized_input = tokenized_input.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized_input,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_output = decoded_output[len(cleaned_input):].strip()
        return predicted_output

    # Load data
    val_df = pd.read_csv(val_data.path)
    print("Shape of val_df:", val_df.shape)

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model.path)
    model = AutoModelForCausalLM.from_pretrained(model.path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
  
    # Predict on sample input
    sample_input = val_df.iloc[0]['input_message_clean']  # Use first validation input
    start_time = time.time()
    predicted_output = predict_single(sample_input, model, tokenizer, device)
    print(f"Sample Input: {sample_input[:100]}...")
    print(f"Predicted Output: {predicted_output}")
    print(f"Prediction Time: {(time.time() - start_time):.2f} seconds")

    # Save prediction
    with open(prediction_output.path, 'w') as f:
        f.write(f"Input: {sample_input}\nPrediction: {predicted_output}")
    print(f"model saved to: {model.path}")
    # Upload to minio
    execution_id = execution_id if execution_id else str(uuid.uuid4())
    object_name = f"{execution_id}/predict/prediction.txt"
    try:
        minio_client.fput_object(bucket_name, object_name, prediction_output.path)
        print(f"Uploaded {object_name} to minio bucket {bucket_name}")
    except S3Error as e:
        print(f"Failed to upload to minio: {e}")

@component(base_image="tarakdhieb7/kubeflow-testing:latest")
def upload_model(model: Output[Model]):
    import subprocess
    subprocess.run(["pip", "install", "minio"], check=True)
    import torch
    import pandas as pd
    from transformers import GPT2Tokenizer, AutoModelForCausalLM
    import time
    from minio import Minio
    from minio.error import S3Error

    # Initialize minio client
    minio_client = Minio(
        "10.106.67.253:9000",
        access_key="minio",  # Replace with your minio access key
        secret_key="minio123",  # Replace with your minio secret key
        secure=False
    )
    # Ensure bucket exists
    bucket_name = "mlpipeline"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    # Upload model files to minio
    
# Define the Pipeline
@dsl.pipeline(name="ml-pipeline")
def ml_pipeline():
    import uuid
    unique_suffix = str(uuid.uuid4())
    execution_id = str(uuid.uuid4())  # Generate a unique execution ID for this pipeline run
    # Step 1: Generate Dataset
    load_op = generate_data(execution_id=execution_id)
    # Step 2: Preprocess Data
    preprocess_op = preprocess_data(input_csv=load_op.outputs["output_csv"], execution_id=execution_id)

    # Step 3: Train Model
#    train_op = train_model(
#      train_data=preprocess_op.outputs["output_train"],
#       val_data=preprocess_op.outputs["output_val"]
#    )

    train_op = train_model(
        train_data=preprocess_op.outputs["output_train"],
        val_data=preprocess_op.outputs["output_val"],
        execution_id=execution_id
    )
    #train_op.set_caching_options(enable_caching=False)
    # Step 4: Evaluate Model
    evaluate_op = evaluate_model(
        val_data=preprocess_op.outputs["output_val"],
        model=train_op.outputs["model_output"],
        execution_id=execution_id
    )

    # Step 5: Predict
    predict_op = predict(
        val_data=preprocess_op.outputs["output_val"],
        model=train_op.outputs["model_output"],
        execution_id=execution_id
    )

if __name__ == "__main__":
    import argparse
    from kfp import Client, compiler
    import kfp_server_api.exceptions as kfp_exceptions
    import uuid

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run and upload Kubeflow pipeline")
    parser.add_argument(
        "--pipeline-exists",
        action="store_true",
        help="Set if the pipeline 'demo_test_distributed' already exists"
    )
    parser.add_argument(
        "--version",
        type=float,
        default=0.1,
        help="Version number for the pipeline (required if --pipeline-exists, e.g., 0.2)"
    )
    args = parser.parse_args()

    
    name_space = "kubeflow-user-example-com"  # Specify the namespace
    endpoint = "http://localhost:8080/pipeline"  # Specify the endpoint
    # initialize a KFPClientManager
    kfp_client_manager = KFPClientManager(
    api_url=endpoint,
    skip_tls_verify=True,

    dex_username="user@example.com",
    dex_password="12341234",

    # can be 'ldap' or 'local' depending on your Dex configuration
    dex_auth_type="local",
)
    client = kfp_client_manager.create_kfp_client()

    
    # Run pipeline with unique run_id to force fresh execution
    try:
        unique_run_id = f"run-{uuid.uuid4()}"
        run_result = client.create_run_from_pipeline_func(
            ml_pipeline,
            arguments={},
            experiment_name="test",
            run_name=unique_run_id,
            namespace="kubeflow-user-example-com",  # Specify the namespace
            service_account="pipeline-runner"  # Specify the service account
        )
        print(f"Started run: {run_result.run_id}")
        print(f"Run details: {endpoint}/#/runs/details/{run_result.run_id}")
    except Exception as e:
        print(f"Failed to start run: {e}")
        exit(1)

    # Compile pipeline
    pipeline_name = "demo_test"
    yaml_file = "kubeflow_pipeline_demo.yaml"
    try:
        compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path=yaml_file)
        print(f"Compiled pipeline to {yaml_file}")
    except Exception as e:
        print(f"Failed to compile pipeline: {e}")
        exit(1)

    # Upload pipeline based on --pipeline-exists flag
    try:
        if args.pipeline_exists:
            if not args.version:
                print("Error: --version required when --pipeline-exists is set")
                exit(1)
            print(f"Uploading version {args.version} for existing pipeline {pipeline_name}")
            try:
                client.upload_pipeline_version(
                    pipeline_package_path=yaml_file,
                    pipeline_version_name=str(args.version),
                    pipeline_name=pipeline_name,
                    description="just for testing"
                )
            except kfp_exceptions.ApiException as api_err:
                if api_err.status == 404:  # Pipeline doesn't exist
                    print(f"Pipeline {pipeline_name} not found, uploading as new pipeline")
                    client.upload_pipeline(
                        pipeline_package_path=yaml_file,
                        pipeline_name=pipeline_name,
                        description="just for testing"
                    )
                else:
                    raise api_err
        else:
            print(f"Uploading new pipeline {pipeline_name}")
            client.upload_pipeline(
                pipeline_package_path=yaml_file,
                pipeline_name=pipeline_name,
                description="just for testing"
            )
    except Exception as e:
        print(f"Failed to upload pipeline: {e}")
        exit(1)