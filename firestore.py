import firebase_admin
from firebase_admin import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector


class FirestoreMan:
    def __init__(self) -> None:
        self.app = firebase_admin.initialize_app()
        self.db = firestore.Client(project="my-gem-in-eye", database="store-esh")
        self.coffee = self.db.collection("stack-exchange-coffee")

    def fetch(self, key):
        print(f"Fetching by {key}");
        doc_ref = self.coffee.document(key)        
        data = doc_ref.get()
        if data.exists:
            print(f"Fetched data: {data.to_dict()}")
        else:
            print("No such document!")
        return data      
    
    def put(self, key, value):
        print(f"Putting at {key}");
        doc_ref = self.coffee.document(key)
        doc_ref.set(value)
        print (f"Stored value at {key}")
    
    def fetchKnn(self, vector, k):
        print(f"Fetching {k} NN for vector");
        vector_query = self.coffee.find_nearest(
            vector_field="embedding_field",
            query_vector=Vector(vector),
            distance_measure=DistanceMeasure.EUCLIDEAN,
            limit=k,
        )
        # print(f"Found {vector_query}")
        docs = vector_query.stream()

        result = []
        for doc in docs:
            docData = doc.to_dict();
            # print(f"doc: {docData}")
            result.append(docData)
            # print(f"{doc.id}, Distance: {doc.get('vector_distance')}")
        
        return result