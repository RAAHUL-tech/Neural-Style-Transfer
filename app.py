from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import os
import tempfile
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import models.transformer as transformer
import models.StyTR as StyTR
import numpy as np
import io
import base64


app = Flask(__name__)
app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "img_db"

mysql = MySQL(app)

# global variables
file_path1 = None
file_path2 = None
relearning = False


def test_transform(size, crop):
    transform_list = []

    if size != 0:
        transform_list.append(transforms.Resize((size,size)))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def generate_image(content_path, style_path):
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str, default = content_path,
                        help='File path to the content image')

    parser.add_argument('--style', type=str, default= style_path,
                        help='File path to the style image, or multiple style \
                        images separated by commas if you want to do style \
                        interpolation or spatial control')
    parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
    parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_4000.pth')
    parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_4000.pth')
    parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_4000.pth')

    parser.add_argument('--style_interpolation_weights', type=str, default="")
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    args = parser.parse_args()

    # Advanced options
    content_size = 512
    style_size = 512
    crop = 'store_true'
    save_ext = '.jpg'
    preserve_color = 'store_true'
    alpha = args.a

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    if args.content:
        content_path = [Path(args.content)]

    if args.style:
        style_path = [Path(args.style)]

    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(args.decoder_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.Trans_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.embedding_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
    network.eval()
    network.to(device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)
    print(content_path)

    content = content_tf(Image.open(content_path[0]).convert("RGB"))
    print("in test", content.shape)
    h, w, c = np.shape(content)
    style = style_tf(Image.open(style_path[0]).convert("RGB"))
    print("in test style", style.shape)

    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)

    with torch.no_grad():
        output, loss_c, loss_s, l_identity1, l_identity2 = network(content, style)
    print(type(output))
    print("output is:", output)
    output = output[0].cpu()
    to_pil_image = transforms.ToPILImage()
    output = to_pil_image(output)
    print("Losses are: Content Fidelity :", loss_c)
    print("Global Effects :", loss_s)
    print("Losses are: id1 Loss :", l_identity1)
    print("Losses are: id2 Loss :", l_identity2)
    return output


@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('index.html')

def image_to_byte_array(image):
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    return img_byte_array

@app.template_filter('b64encode')
def b64encode_filter(s):
    return base64.b64encode(s).decode('utf-8')


@app.route('/upload', methods=['POST'])
def upload():
    if 'content' not in request.files or 'style' not in request.files:
        return "No Images Uploaded!"

    content = request.files['content']
    style = request.files['style']

    if content.filename == '' or style.filename == '':
        return "No Image Selected!"

    temp_dir = tempfile.gettempdir()
    global file_path1
    global file_path2
    file_path1 = os.path.join(temp_dir, content.filename)
    file_path2 = os.path.join(temp_dir, style.filename)
    content.save(file_path1)
    style.save(file_path2)
    generated = generate_image(file_path1, file_path2)
    image_data = image_to_byte_array(generated).read()
    with open(file_path1, 'rb') as f:
        data1 = f.read()
    with open(file_path2, 'rb') as f:
        data2 = f.read()
    '''
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO images (content, style) VALUES (%s, %s)", (data1, data2))
    mysql.connection.commit()
    cur.close()
    '''
    return render_template('generate.html', image_data=image_data)


def perform_reinforcement_learning(rating):
    global file_path1
    global file_path2
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str, default =file_path1,help='File path to the content image')
    parser.add_argument('--style', type=str, default =file_path2,help='File path to the style image')
    parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
    parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_4000.pth')
    parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_4000.pth')
    parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_4000.pth')
    parser.add_argument('--a', type=float, default=1.0)
    args = parser.parse_args()
    # Set device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # Load model
    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()
    # Load pre-trained weights
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.decoder_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.Trans_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.embedding_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    # Create network
    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
    network.train()
    network.to(device)
    network = nn.DataParallel(network, device_ids=[0, 1])
    # Load content and style images
    content_tf = test_transform(256, False)
    style_tf = test_transform(256, False)

    content_image = content_tf(Image.open(args.content).convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)
    style_image = style_tf(Image.open(args.style).convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)

    # Hyperparameters
    content_weight = 8.0
    style_weight = 10.0
    learning_rate = 0.001

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': network.module.transformer.parameters()},
        {'params': network.module.decode.parameters()},
        {'params': network.module.embedding.parameters()}
    ], lr=learning_rate)

    print(content_image.shape)
    print(style_image.shape)
    # Forward pass
    output, loss_c, loss_s, l_identity1, l_identity2 = network(content_image, style_image)

    # Compute total loss
    loss = content_weight * loss_c + style_weight * loss_s + (l_identity1 * 70) + (l_identity2 * 1)
    # adding penalty
    loss = loss * int(rating)
    print("Loss",loss)
    # Backward pass
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    # Save updated model parameters
    state_dict = network.module.transformer.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, 'reinforment_model/updated_transformer.pth')
    state_dict = network.module.decode.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, 'reinforment_model/updated_decoder.pth')
    state_dict = network.module.embedding.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, 'reinforment_model/updated_embedding.pth')

    # Print losses
    print("Losses: Content Fidelity:", loss_c)
    print("Global Effects:", loss_s)
    print("Identity 1 Loss:", l_identity1)
    print("Identity 2 Loss:", l_identity2)


@app.route('/rate', methods=['POST'])
def rate():
    if 'rating' not in request.form:
        return "No Rating Selected!"
    rating = request.form['rating']
    '''
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO ratings (rate) VALUES (%s)", (rating))
    mysql.connection.commit()
    cur.close()
    '''
    global relearning
    if relearning is True:
        perform_reinforcement_learning(rating)
    return "Thank you for your rating!"

if __name__ == '__main__':
    app.run(debug=True)