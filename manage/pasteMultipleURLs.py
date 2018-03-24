import pyperclip, sys, math


def geneUrl(url, numPic):
    num = math.ceil(numPic / 4)
    array = url.split(".")
    init = array[2]
    urlList = []
    urlList.append(url)
    for i in range(1, num):
        mod = "".join([init, "_", str(i+1)])
        newUrl = [array[0], array[1], mod, array[3]]
        urlList.append(".".join(newUrl))
    print("\n".join(urlList))
    pyperclip.copy("\n".join(urlList))


list = ['https://www.meitulu.com/item/', sys.argv[1], '.html']
# list = ['http://www.192tt.com/gq/youmi/ym', sys.argv[1], '.html#p']
url = "".join(list)
pic = int(sys.argv[2])
geneUrl(url, pic)
