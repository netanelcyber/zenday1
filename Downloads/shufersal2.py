from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import requests
import re

app = Flask(__name__)
api = Api(app, version='1.0', title='Shufersal API',
          description='API לזחילה משופרסל: קטגוריות, קישורים ומוצרים')

ns_products = api.namespace('products', description='שליפת מוצרים')
ns_categories = api.namespace('categories', description='שליפת קטגוריות')
ns_links = api.namespace('parse-link', description='חילוץ category_id מקישור שופרסל')

product_model = api.model('Product', {
    'name': fields.String,
    'price': fields.Float,
    'sku': fields.String,
    'image': fields.String
})

product_query = api.model('ProductQuery', {
    'category_id': fields.Integer(required=True),
    'take': fields.Integer(default=20),
    'skip': fields.Integer(default=0)
})

link_query = api.model('LinkQuery', {
    'url': fields.String(required=True, description='קישור לשופרסל')
})

category_model = api.model('Category', {
    'id': fields.Integer,
    'name': fields.String,
    'path': fields.String
})


def fetch_products(category_id, take=20, skip=0):
    url = f"https://www.shufersal.co.il/online/api/product-category/{category_id}?take={take}&skip={skip}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return [{
        'name': p.get('Name'),
        'price': p.get('Price', {}).get('Price'),
        'sku': p.get('Sku'),
        'image': p.get('Image')
    } for p in r.json().get('Products', []) if p.get('Name')]


def fetch_categories():
    url = "https://www.shufersal.co.il/online/api/category-tree"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    tree = r.json()

    result = []

    def walk(nodes, path=""):
        for node in nodes:
            if node.get("IsCategory"):
                current_path = f"{path}/{node['Description']}".strip("/")
                result.append({
                    'id': node.get("CategoryID"),
                    'name': node.get("Description"),
                    'path': current_path
                })
                if node.get("ChildCategories"):
                    walk(node["ChildCategories"], current_path)

    walk(tree)
    return result


def extract_category_id_from_link(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    match = re.search(r'"CategoryID":(\d+)', r.text)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("לא נמצא category_id בקישור")


@ns_products.route('/')
class ProductList(Resource):
    @api.expect(product_query)
    @api.marshal_list_with(product_model)
    def post(self):
        """שליפת מוצרים לפי קטגוריה"""
        data = request.get_json()
        return fetch_products(data['category_id'], data.get('take', 20), data.get('skip', 0))


@ns_categories.route('/')
class CategoryList(Resource):
    @api.marshal_list_with(category_model)
    def get(self):
        """החזרת כל הקטגוריות"""
        return fetch_categories()


@ns_links.route('/')
class LinkParser(Resource):
    @api.expect(link_query)
    def post(self):
        """חילוץ category_id מקישור שופרסל"""
        data = request.get_json()
        try:
            cat_id = extract_category_id_from_link(data['url'])
            return {'category_id': cat_id}
        except Exception as e:
            api.abort(400, str(e))


if __name__ == '__main__':
    app.run(debug=True)

