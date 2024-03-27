<h2>This is a python library that outputs an image with style transfer applied only to the pigeons in the photograph.</h2>
<ul>
  <li>Input: A photo of a pigeon and an image to apply style changes.<br>
  <img src="/system_img_e_g/input_sample.jpg" height="200"><img src="/system_img_e_g/style_sample.jpg" height="200"></li><br>
  <li>Output: An image with style transformation applied only to the pigeons in the image.
  <img src="/system_img_e_g/output_sample.png" width="600"></li>
</ul>

<h3>Steps</h3>
<ol>
  <li>Extract and crop only the pigeon in the image.</li><br>
  <li>Apply style transfer to the cropped image of the pigeon.</li><br>
  <li>Combine the stylized pigeon with the original image.</li>
</ol>

<h3>How to use</h3>
<ol>
  <li>Enter the following command in the command line: <code>pip install git+https://github.com/ritsuki-i/Creating-stylish-pigeons</code></li><br>
  <li>If you want to update, enter the following command in the command line: <code>pip install git+https://github.com/ritsuki-i/Creating-stylish-pigeons -U</code></li><br>
  <li>Write <code>import stylechange</code> in the Python file.</li><br>
  <li>Write <code>stylechange.stylechange(input_path, style_path)</code> in the Python file.("input_path" is the path to the photo of the creature whose style will be changed, and "style_path" is the path to the image of the style.)</li>
</ol>