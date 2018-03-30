
import random
import numpy as np
import chainer.datasets.image_dataset
import chainer.links as L
import chainer.functions as F
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','gif','tif','tiff'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Network(chainer.Chain):
    def __init__(self, n_class):
        super(Network, self).__init__(
            conv1=L.Convolution2D(None, 64, 3, pad=1),
            conv2=L.Convolution2D(None, 128, 3, pad=1),
            conv3=L.Convolution2D(None, 256, 1),
            conv4=L.Convolution2D(None, 512, 1),
            conv5=L.Convolution2D(None, 1024, 1),
            conv6=L.Convolution2D(None, n_class, 1),

            norm1=L.BatchNormalization(64),
            norm2=L.BatchNormalization(128),
            norm3=L.BatchNormalization(256),
            norm4=L.BatchNormalization(512),
            norm5=L.BatchNormalization(1024),
            )

    def __call__(self, x, test=False):
        h1 = F.relu(F.max_pooling_2d(self.norm1(self.conv1(x), test=test), 2))
        h2 = F.relu(F.max_pooling_2d(self.norm2(self.conv2(h1), test=test), 2))
        h3 = F.relu(self.norm3(self.conv3(h2), test=test))
        h4 = F.relu(self.norm4(self.conv4(h3), test=test))
        h5 = F.relu(self.norm5(self.conv5(h3), test=test))
        h6 = self.conv6(h5)
        y = F.average_pooling_2d(h6, (h6.shape[2], h6.shape[3]))
        y = y[:, :, 0, 0]
        return y


@app.route('/')
def index():
    message = '<img src="/static/top.png">'
    return render_template('index.html', message=message)


@app.route('/post', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = 'uploads/' + \
                str(random.randint(0, 10000)) + '.' + \
                file.filename.rsplit('.', 1)[1].lower()
            file.save(filename)
            name = 'static/tmp/' + str(random.randint(0, 10000)) + '.png'
            list = open('assets/list_{}.txt'.format(request.form['type']), 'r')
            font_dirs = []
            for x in list:
                font_dirs.append(x.rstrip("\n"))
            list.close()
            n_class = len(font_dirs)
            model = Network(n_class)
            chainer.serializers.load_npz(
                'assets/model_{}'.format(request.form['type']), model)
            img = Image.open(filename)
            img.resize((64, 64), Image.LANCZOS).convert('L').save(name)
            f = chainer.datasets.image_dataset._read_image_as_array(
                name, np.float32)
            x = f.reshape((1, 1) + f.shape)
            label = F.softmax(model(x, test=True)).data
            result = '<h2>可能性のあるフォント</h2>'
            predicts = np.argsort(label[0])[::-1]
            for i, predict in enumerate(predicts[:10]):
                if label[0, predict] * 100 >= 1:
                    result += font_dirs[predict].split('/')[1] + '<br>'
            message = '<img src="/' + name + '" style="width: 128px;""> ' + result
            return render_template('index.html', message=message)
        message = '画像形式が正しくありません(png,jpg,jpegのみです)'
        return render_template('index.html', message=message)
    else:
        message = '<img src="/static/top.png">'
        return render_template('index.html', message=message)
        

if __name__ == '__main__':
    app.run(host='0.0.0.0')