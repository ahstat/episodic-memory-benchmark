def upload_ftuning_input(client, ftuning_input_filepath):
    client.files.create(file=open(ftuning_input_filepath, "rb"), purpose="fine-tune")
    print(f"File {ftuning_input_filepath} uploaded")
    return 0

def retrieve_fileid(client, jsonl_filename):
    file_listing = client.files.list()
    print(file_listing)
    corresponding_ids =[x.id for x in file_listing if x.filename == jsonl_filename]
    if len(corresponding_ids) == 0:
        print(jsonl_filename)
        raise ValueError('file not found, please first upload it with the `answering_parameters["ftuning_need_upload"] = True` argument')
    if len(corresponding_ids) > 1:
        print('length of the matching jsonl should be exactly one?')
    return corresponding_ids[0]