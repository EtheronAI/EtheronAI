from flask import Blueprint, request, jsonify
import hashlib
import time
import json

# Create blueprint
blockchain_bp = Blueprint('blockchain', __name__)

class Blockchain:
    def __init__(self):
        self.chain = []
        self.nodes = set()
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'proof': proof,
            'previous_hash': previous_hash,
            'data': "Block Data"
        }
        self.chain.append(block)
        return block

    def add_node(self, address):
        self.nodes.add(address)

    def proof_of_work(self, previous_proof):
        proof = 0
        while self.valid_proof(previous_proof, proof) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(previous_proof, proof):
        guess = f'{previous_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    def get_last_block(self):
        return self.chain[-1]

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

# Initialize blockchain
blockchain = Blockchain()

@blockchain_bp.route('/mine', methods=['GET'])
def mine_block():
    """
    Mine Block API
    ---
    responses:
      200:
        description: New block information
        schema:
          type: object
          properties:
            message:
              type: string
            index:
              type: integer
            timestamp:
              type: number
            proof:
              type: integer
            previous_hash:
              type: string
    """
    last_block = blockchain.get_last_block()
    proof = blockchain.proof_of_work(last_block['proof'])
    block = blockchain.create_block(proof, blockchain.hash(last_block))
    response = {
        'message': "New Block Mined",
        'index': block['index'],
        'timestamp': block['timestamp'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash']
    }
    return jsonify(response), 200

@blockchain_bp.route('/nodes/add', methods=['POST'])
def add_node():
    """
    Add Node API
    ---
    parameters:
      - name: nodes
        in: body
        type: array
        required: true
        description: List of node addresses
    responses:
      201:
        description: Nodes added successfully
        schema:
          type: object
          properties:
            message:
              type: string
            total_nodes:
              type: array
    """
    values = request.get_json()
    nodes = values.get('nodes')
    for node in nodes:
        blockchain.add_node(node)
    response = {
        'message': 'New nodes added',
        'total_nodes': list(blockchain.nodes)
    }
    return jsonify(response), 201

@blockchain_bp.route('/chain', methods=['GET'])
def get_chain():
    """
    Get Blockchain API
    ---
    responses:
      200:
        description: Blockchain data
        schema:
          type: object
          properties:
            chain:
              type: array
            length:
              type: integer
    """
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain)
    }
    return jsonify(response), 200