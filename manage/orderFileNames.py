import cv2, os, numpy as np


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8), -1)     # successful read if path has Chinese
    return cv_img


if __name__ == '__main__':
    path = 'C:\\Users\\a\\Google 云端硬盘\\图片\\Yang'
    folders = os.listdir(path)
    for folder in folders:
        if os.path.isdir(os.path.join(path, folder)):
            imageFiles = os.listdir(os.path.join(path, folder))
            try:
                sortedFiles = sorted(imageFiles, key=lambda file: int(file.split('.')[0]))
            except:
                if imageFiles[0].split('.')[0] == 'h1':
                    print("%s has been renamed" % folder)
                    continue
                else:
                    sortedFiles = imageFiles
            h = 0
            w = 0
            for file in sortedFiles:
                src = os.path.join(path, folder, file)
                img = cv_imread(os.path.join(path, folder, file))
                if(img.shape[0] > img.shape[1]):
                    h += 1
                    newName = os.path.join(folder, 'h'+str(h)+'.jpg')
                    dst = os.path.join(path, folder, 'h'+str(h)+'.jpg')
                else:
                    w += 1
                    newName = os.path.join(folder, 'w' + str(w) + '.jpg')
                    dst = os.path.join(path, folder, 'w'+str(w)+'.jpg')
                print("the file name is %s, rename as %s" % (os.path.join(folder, file), newName))
                os.rename(src, dst)
        else:
            continue
