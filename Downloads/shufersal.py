from flask import Flask, request, jsonify

from flask_restx import Api, Resource, fields

from selenium import webdriver

from selenium.webdriver.common.by import By

from selenium.webdriver.common.keys import Keys

import time



app = Flask(__name__)

api = Api(app, version='1.0', title='Shufersal Search API',

          description='API לחיפוש מוצרים באתר שופרסל באמצעות Selenium')



ns = api.namespace('search', description='חיפוש מוצרים')



search_model = api.model('SearchQuery', {

    'query': fields.String(required=True, description='מילת החיפוש')

})





@ns.route('/')

class ProductSearch(Resource):

    @api.expect(search_model)

    def post(self):

        """חפש מוצרים בשופרסל לפי מחרוזת חיפוש"""

        data = request.get_json()

        query = data['query']



        # הפעלת דפדפן

        options = webdriver.ChromeOptions()

        options.add_argument('--headless')  # ללא GUI

        driver = webdriver.Chrome(options=options)



        driver.get("https://www.shufersal.co.il")

        time.sleep(5)



        try:

            search_box = driver.find_element(By.ID, "js-site-search-input")

            search_box.send_keys(query)

            search_box.send_keys(Keys.RETURN)

            time.sleep(5)



            products = driver.find_elements(By.CLASS_NAME, "product-title")

            results = [p.text for p in products[:10] if p.text.strip()]

        except Exception as e:

            driver.quit()

            return {'error': str(e)}, 500



        driver.quit()

        return jsonify(results)





if __name__ == '__main__':

    app.run(debug=True)


