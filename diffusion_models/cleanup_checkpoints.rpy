def do():
    #Go into a results folder and run this. It will delete all but the latest checkpoint with your confirmation.    
    files=get_all_files(sort_by="number", file_extension_filter="pt", relative=True)
    del_files=files[:-1]
    keep_files=files[-1:]
    print('Pwd: '+get_current_directory())
    os.system('du -sh')
    print('Files marked for deletion: '+fansi(del_files,'red'))
    print('Keeping files: '+fansi(keep_files,'green'))
    if input_yes_no('Does this look good to you?'):
        delete_files(files)
        print('Done - deleted %i files'%len(del_files))
    else:
        print("Ok, cancelled.")
    os.system('du -sh')
