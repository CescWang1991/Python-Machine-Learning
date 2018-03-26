import sys, urllib.request

urlPre = 'https://mtl.ttsqgs.com/images/img/'
urlSet = []
num = int(sys.argv[2])
for i in range(num):
    url = ''.join((urlPre, str(sys.argv[1]), '/', str(i+1), '.jpg'))
    urlSet.append(url)
try:
    imgUrl = urlSet[0]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    req = urllib.request.Request(url=imgUrl, headers=headers)
    # urllib.request.urlopen(req).read()
    urllib.request.urlretrieve(req, '1.jpg')
except Exception as exc:
    print('There was a problem: %s' % (exc))
