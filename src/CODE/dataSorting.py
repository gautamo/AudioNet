#collect and split data into training, validation, and test sets

import convFilter
import os, shutil
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import split_folders

def countClasses(data_root, debugContent=False): #counts the classes and number of data from root
    root = os.path.join(data_root, "all/all")
    class_count = 0
    data_count = 0
    
    for dataClass in os.listdir(root):
        class_count +=1
        for dataItem in os.listdir(root + "/" + dataClass):
            data_count += 1
        
        if debugContent:
            img = cv2.imread(root + "/" + dataClass + "/" + dataItem, cv2.IMREAD_GRAYSCALE)
            strImg = str(','.join(str(item) for innerlist in img for item in innerlist))
            print("DataClass: ", dataClass)
            print("Img String", strImg)
            
    print(f"Total Classes: {class_count}")
    print(f"Total Data: {data_count}")
    
def initiateData(data_root): #CLEARS ALL DATA IN DATA_ROOT FOLDER
    for item in os.listdir(data_root):
        assert item in ["all", "train", "validation", "test", ""] #safety check to make sure root is correct
        shutil.rmtree(os.path.join(data_root,item)) #clears all items in root
    all_dir = os.path.join(data_root, "all")
    os.mkdir(all_dir)
    all_root = os.path.join(all_dir, "all")
    os.mkdir(all_root) #creates all/all in root dir
    
def populateData(filtered_performances, data_root, save=True):
    """
    Slices spectrograms in filtered_performances and saves images in all/all folder with class as folder name
    """
    if save:
        initiateData(data_root) #clear data in root folder
        
    all_root = os.path.join(data_root, "all/all")
    class_count = defaultdict(int)
    data_count = 0
    for piecenum in range(len(filtered_performances)):
        print(f"Piece {piecenum} of {len(filtered_performances)}")
        
        piece = filtered_performances[piecenum]
        try:
            performance = piece.load_performance(piece.available_performances[0], require_audio=False)
            spectrogram = performance.load_spectrogram()
        except Exception as e:
            print(f"EXCEPTION at Piece {piecenum}: {e}")
            continue
            
        for slice in range(spectrogram.shape[1]):
            try:
                trueVal = str(int(''.join(map(str, convFilter.getNvec(slice, performance))), 2))
                trueSpec = convFilter.getSpectrogram(slice, performance)
                
                if trueVal in class_count:
                    class_count[trueVal] += 1
                else:
                    class_count[trueVal] = 1
                data_count += 1
                
                if save:
                    addImageToDirectory(trueSpec, f"img{class_count[trueVal]}.png", trueVal, all_root)
                
            except IndexError as e:
                print(f"INDEXERROR: PieceNum: {piecenum}, Slice: {slice}, Message: {e}")    
    
    print(f"Total Classes: {len(class_count)}")
    print(f"Total Data: {data_count}")
            
def addImageToDirectory(image, imageName, folder, root):
    """
    Adds image to directory specified
    """
    class_root = os.path.join(root, folder)
    if os.path.isdir(class_root):
        cv2.imwrite(os.path.join(class_root, imageName), image)
    else:
        try:  
            os.mkdir(class_root)  
            cv2.imwrite(os.path.join(class_root, imageName), image)
        except OSError as error:  
            print(error)

def divideDataIntoTrainValTestSets(data_root, train=.6, val=.2, test=.2):
    """
    Distributes data into train, validation, and test sets from all/all folder
    """
    all_root = os.path.join(data_root, "all/all")
    
    assert train + val + test == 1
    
    split_folders.ratio(all_root, output=data_root, seed=1337, ratio=(train, val, test)) # default values
    os.rename(os.path.join(data_root, "val"), os.path.join(data_root, "validation"))
    
    temp = "Temp"
    os.mkdir(os.path.join(data_root, "train" + temp))
    os.mkdir(os.path.join(data_root, "validation" + temp))
    os.mkdir(os.path.join(data_root, "test" + temp))
    
    for item in ["train", "validation", "test"]:
        dest = shutil.move(os.path.join(data_root, item), os.path.join(data_root, item + temp))
        os.rename(os.path.join(data_root, item + temp), os.path.join(data_root, item))
        
def genNPY(filtered_performances, dataRoot: str):
    """
    Generate the numpy files for all compiled spectrogram and midi data.
    Requires npyversion folder to be made inside of data_root foler.
    filtered_performances: all performances that have spectrogram and midi data available
    dataRoot(str): path to data_root folder.
    """
    piece = filtered_performances[0]
    performance = piece.load_performance(piece.available_performances[0], require_audio=False)
    all_spectro = performance.load_spectrogram() #loads first spectrogram numpy array to concatenate data to
    all_midi = performance.load_midi_matrix() #loads first midi numpy array to concatenate data to

    for piece_num in range(1, len(filtered_performances)):
        if(piece_num % 30 == 0): print('Piece %d of %d' % (piece_num,len(filtered_performances)))
        temp_piece = filtered_performances[piece_num]
        temp_performance = temp_piece.load_performance(temp_piece.available_performances[0], require_audio=False)
        try:
            temp_spectro = temp_performance.load_spectrogram() #loads next spectrogram data
            temp_midi = temp_performance.load_midi_matrix() #loads next midi data
        except:
            continue #no spectrogram data or midi data present for current performance
        if(temp_spectro.shape[1] > temp_midi.shape[1]): #spectrogram data is longer length than midi data
            all_spectro = np.concatenate((all_spectro, temp_spectro[:,:temp_midi.shape[1]]), axis=1)
            all_midi = np.concatenate((all_midi, temp_midi), axis=1)
        else: #midi data is longer length than spectrogram data
            all_spectro = np.concatenate((all_spectro, temp_spectro), axis=1)
            all_midi = np.concatenate((all_midi, temp_midi[:,:temp_spectro.shape[1]]), axis=1)

    np.save(dataRoot+'/npyversion/all_spectro.npy', all_spectro) #saving spectrogram numpy file
    np.save(dataRoot+'/npyversion/all_midi.npy', all_midi) #saving midi numpy file
    return all_spectro, all_midi

def genSplitNPY(spectro, midi, dataRoot):
    """
    Generate the numpy files for all compiled spectrogram and midi data split into training,validation, and test.
    Split is 60% training, 20% validation, 20% test.
    Requires npyversion folder to be made inside of data_root foler.
    spectro: numpy array filled with all spectrogram data
    midi: numpy array filled with all midi data
    dataRoot(str): path to data_root folder.
    """
    length=spectro.shape[1]
    idx = np.random.choice(range(length), length, replace=False)
    train = idx[:918259]
    val = idx[918259:1224345]
    test = idx[1224345:length] #60% train, 20% val, 20% test

    ################
    #Training Split#
    ################
    train_spectro = np.empty([92,0], dtype=np.float32) #empty training spectrogram array to concatenate to
    train_midi = np.empty([128,0], dtype=np.uint8) #empty training midi array to concatenate to
    for batch in range(92): #split data into batches to speed up concatenation
        if batch != 91: batch_len = 10000 
        else: batch_len = 8259 #very last batch of data
        temp_spectro1 = np.empty([92,0], dtype=np.float32)
        temp_midi1 = np.empty([128,0], dtype=np.uint8)
        for idx in range(batch_len):
            index = (batch*10000)+idx
            temp_spectro2 = spectro[:,train[index]].reshape((92,1))
            temp_midi2 = midi[:,train[index]].reshape((128,1))
            temp_spectro1 = np.concatenate((temp_spectro1, temp_spectro2), axis=1) #append temp spectrogram data to training
            temp_midi1 = np.concatenate((temp_midi1, temp_midi2), axis=1) #append temp midi data to training
        train_spectro = np.concatenate((train_spectro, temp_spectro1), axis=1)
        train_midi = np.concatenate((train_midi, temp_midi1), axis=1)
        print('Batch %d complete' % batch)
    np.save(dataRoot+'/npyversion/train_spectro.npy', train_spectro) #save training split spectrogram numpy file
    np.save(dataRoot+'/npyversion/train_midi.npy', train_midi) #save training split midi numpy file
    print('Training split complete')
    
    ##################
    #Validation Split#
    ##################
    val_spectro = np.empty([92,0], dtype=np.float32)
    val_midi = np.empty([128,0], dtype=np.uint8)
    for batch in range(31):
        if batch != 30: batch_len = 10000
        else: batch_len = 6086
        temp_spectro1 = np.empty([92,0], dtype=np.float32)
        temp_midi1 = np.empty([128,0], dtype=np.uint8)
        for idx in range(batch_len):
            index = (batch*10000)+idx
            temp_spectro2 = spectro[:,val[index]].reshape((92,1))
            temp_midi2 = midi[:,val[index]].reshape((128,1))
            temp_spectro1 = np.concatenate((temp_spectro1, temp_spectro2), axis=1)
            temp_midi1 = np.concatenate((temp_midi1, temp_midi2), axis=1)
        val_spectro = np.concatenate((val_spectro, temp_spectro1), axis=1)
        val_midi = np.concatenate((val_midi, temp_midi1), axis=1)
        print('Batch %d complete' % batch)
    np.save(dataRoot+'/npyversion/val_spectro.npy', val_spectro)
    np.save(dataRoot+'/npyversion/val_midi.npy', val_midi)
    print('Validation split complete')
    
    ############
    #Test Split#
    ############
    test_spectro = np.empty([92,0], dtype=np.float32)
    test_midi = np.empty([128,0], dtype=np.uint8)
    for batch in range(31):
        if batch != 30: batch_len = 10000
        else: batch_len = 6086
        temp_spectro1 = np.empty([92,0], dtype=np.float32)
        temp_midi1 = np.empty([128,0], dtype=np.uint8)
        for idx in range(batch_len):
            index = (batch*10000)+idx
            temp_spectro2 = spectro[:,test[index]].reshape((92,1))
            temp_midi2 = midi[:,test[index]].reshape((128,1))
            temp_spectro1 = np.concatenate((temp_spectro1, temp_spectro2), axis=1)
            temp_midi1 = np.concatenate((temp_midi1, temp_midi2), axis=1)
        test_spectro = np.concatenate((test_spectro, temp_spectro1), axis=1)
        test_midi = np.concatenate((test_midi, temp_midi1), axis=1)
        print('Batch %d complete' % batch)
    np.save(dataRoot+'/npyversion/test_spectro.npy', test_spectro)
    np.save(dataRoot+'/npyversion/test_midi.npy', test_midi)
    print('Test split complete')

def loadAllData(dataRoot: str):
    """
    Loads all spectrogram and all midi data from numpy files.
    dataRoot(str): path to data_root folder.
    """
    return np.load(dataRoot+'/npyversion/all_spectro.npy'), np.load(dataRoot+'/npyversion/all_midi.npy')

def loadSplitData(dataRoot: str):
    """
    Loads all spectrogram and all midi split data from numpy files.
    dataRoot(str): path to data_root folder.
    """
    train = (np.load(dataRoot+'/npyversion/train_spectro.npy'), np.load(dataRoot+'/npyversion/train_midi.npy'))
    val = (np.load(dataRoot+'/npyversion/val_spectro.npy'), np.load(dataRoot+'/npyversion/val_midi.npy'))
    test = (np.load(dataRoot+'/npyversion/test_spectro.npy'), np.load(dataRoot+'/npyversion/test_midi.npy'))
    return train, val, test
        
if __name__ == "__main__":
    DATA_ROOT_MSMD = '/Users/gbanuru/PycharmProjects/HACKUCI/msmd_aug_v1-1_no-audio/' # path to MSMD data set
    data_root = "/Users/gbanuru/PycharmProjects/HACKUCI/msmd/tutorials/data_root" # path to our created dataset  
    
    #filtered_performances = convFilter.filteredData(DATA_ROOT_MSMD) #creates a list with piece objects
    #print(f"All pieces: {len(filtered_performances)}")
    #populateData(filtered_performances[:], data_root, save = False)
    #divideDataIntoTrainValTestSets(data_root)
    
    countClasses(data_root)
    
    print("Done")