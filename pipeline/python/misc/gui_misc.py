
# =====================================================================================
# Ask yes/no question
# =====================================================================================

def askUser(title, prompt):
    from tkinter import Tk, filedialog, messagebox
    root = Tk()
    root.withdraw()

    r = messagebox.askyesno(title, prompt)

    return r

# =====================================================================================
# Ask for input file
# =====================================================================================

def askUserForFile():

    # Ask for file
    from tkinter import Tk, filedialog, messagebox
    root = Tk()
    root.withdraw()

    selectedFile = filedialog.askopenfilename()

    return selectedFile

# =====================================================================================
# Ask for input directories
# =====================================================================================

def askUserForDirectory(findUnprocessedRecordings, returnFilenames = False, forceOverwrite=False):

    # Ask for the directory
    from tkinter import Tk, filedialog, messagebox
    root = Tk()
    root.withdraw()

    selectedDir = filedialog.askdirectory()

    fnamesOverwrite   = findUnprocessedRecordings(selectedDir, overwrite=True)
    fnamesNoOverwrite = findUnprocessedRecordings(selectedDir, overwrite=False)

    nRec = len(fnamesNoOverwrite)
    nAll = len(fnamesOverwrite)

    # If there are no files regardless, let user know immediately instead of asking for overwrite decision
    OVERWRITE_RECORDING = forceOverwrite

    if nAll == 0:
        messagebox.showwarning("No files to process.", "No files found to process. Exiting...")
    elif nAll != nRec and not forceOverwrite:
        # Ask if we should reprocess/overwrite
        OVERWRITE_RECORDING = messagebox.askyesno('Overwrite?',
                                                  'Do you want to re-process and overwrite already-processed recordings?')

    # Which filenames to return, if asked for
    fnamesToReturn = fnamesOverwrite if OVERWRITE_RECORDING else fnamesNoOverwrite

    # Ask user if they're sure about overwriting/reprocessing, because this decision might have major influences on
    # run time
    if OVERWRITE_RECORDING and not forceOverwrite:
        OVERWRITE_RECORDING = messagebox.askyesno('Overwrite: Are you sure?',
            'Are you sure you want to re-process and overwrite already-processed recordings? ' + \
            'This will process all {} instead of only {} recordings.'.format(
            nAll, nRec))

        if OVERWRITE_RECORDING:
            nRec = nAll

    # Check that there are files left
    if nRec == 0 and not nAll == 0:
        messagebox.showwarning("No files to process.", "No files found to process. Exiting...")

    if returnFilenames:
        return selectedDir, OVERWRITE_RECORDING, nRec, fnamesToReturn
    else:
        return selectedDir, OVERWRITE_RECORDING, nRec