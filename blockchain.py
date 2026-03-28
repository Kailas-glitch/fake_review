import hashlib
import json
from time import time
from datetime import datetime

class Blockchain:

    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # ← readable format
            'proof': proof,
            'previous_hash': previous_hash,
            'data': None
        }
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def proof_of_work(self, previous_proof):
        new_proof = 1

        while True:
            hash_operation = hashlib.sha256(
                str(new_proof**2 - previous_proof**2).encode()
            ).hexdigest()

            if hash_operation[:4] == '0000':
                return new_proof

            new_proof += 1

    def add_review(self, review_data):
        previous_block = self.get_previous_block()
        proof = self.proof_of_work(previous_block['proof'])
        previous_hash = self.hash(previous_block)

        block = self.create_block(proof, previous_hash)
        block['data'] = review_data
        self.chain[-1]['data']=review_data

        return block

    def is_chain_valid(self):
        previous_block = self.chain[0]

        for index in range(1, len(self.chain)):
            block = self.chain[index]

            if block['previous_hash'] != self.hash(previous_block):
                return False

            previous_proof = previous_block['proof']
            proof = block['proof']

            hash_operation = hashlib.sha256(
                str(proof**2 - previous_proof**2).encode()
            ).hexdigest()

            if hash_operation[:4] != '0000':
                return False

            previous_block = block

        return True