from flask import Flask, request, render_template
import app_utils

flask_api = Flask(__name__)

@flask_api.route('/', methods=['GET'])
def home():
#    return render_template('/home/sindhu/iisc-capstone/group4capstone-code/src/ui/imageCaptioning.html')
   return render_template('imageCaptioning.html')

@flask_api.route('/inference', methods=['POST'])
def inference():
    print('here')
    data = request.get_json()
    
    print(data)
    file_path = data['file_path']
    caption = app_utils.inference_function(file_path)
    corrected_caption = app_utils.grammar_correction(caption)
    print(caption)
    return {"caption": corrected_caption}
 

if __name__ == '__main__':
    flask_api.run(debug=True) 