import html
import os
import json
import re
import string
import sys
import uuid
from collections import Counter
from functools import partial
from typing import Union

import gensim.downloader as api
import hdbscan
import nltk
import numpy as np
import pandas as pd
import requests
import spacy
from gensim.models import KeyedVectors
from gnewsclient import gnewsclient
from google.api_core.exceptions import InvalidArgument
from google.cloud import language_v1
from google.cloud.language_v1 import AnalyzeEntitiesRequest
from loguru import logger
from pydantic import BaseModel
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from core.config import IS_TEST_MODE
from core.messages import NO_VALID_PAYLOAD, NOT_FOUND_MODEL
from models.payload import EntityPayload, BingSearchPayload, TwitterTrendPayload, GoogleNewsPayload
from models.prediction import EntityResult, RawEntityResult, RefineEntityResult, ClassificationResult, SentimentResult, \
    DistanceMatrixResult
from services.vendor.google_nlp import GoogleNlpWrapper
from services.vendor.twitter_trend import TwitterTrendWrapper
from services.graph.helper import PlotlyGraphClient
from utils import helper
from db import dynamodb


class LanguageModel(object):
    MAX_TEXT_SIZE = 30 * 1024
    RESULT_UNIT_FACTOR = 100000
    BING_API = "https://api.bing.microsoft.com/v7.0/search"
    punctuations = dict((ord(char), None) for char in string.punctuation)

    def __init__(self, doc2vec_path: str, word2vec_path: str, db: dynamodb.DbLogger = None):
        self.doc2vec_path = doc2vec_path
        self.word2vec_path = word2vec_path
        self._load_local_model()
        self.db = db
        self.google_nlp = GoogleNlpWrapper()
        self.twitter_trend = TwitterTrendWrapper()
        # self.google_news = GoogleNewsWrapper()

    def _load_local_model(self):

        self.spacy_nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
        self.vectorizer = self.tfidf_vectorizer()
        self.scaler = StandardScaler()
        # self.stop_list =spacy.lang.pt.stop_words.STOP_WORDS

        # if os.path.isfile(self.doc2vec_path):
        #     self.doc2vec = Doc2Vec.load(self.doc2vec_path, mmap='r')
        # else:
        #     logger.debug(f"Not found doc2vec model path: {self.doc2vec_path}")
        #     self.doc2vec = None
        if IS_TEST_MODE:
            # for quickly testing
            self.word2vec = api.load("glove-wiki-gigaword-50")
        elif os.path.isfile(self.word2vec_path):
            self.word2vec = KeyedVectors.load_word2vec_format(self.word2vec_path)
        else:
            logger.debug(f"Not found word2vec model path: {self.word2vec_path}")
            self.word2vec = None

    def __clean_text(self, text):
        # return text.replace('.', "").replace('-', ' ')
        return " ".join(nltk.word_tokenize(text.lower().translate(self.punctuations)))

    def text_norm(self, text: str, stemmer: nltk.stem.porter.PorterStemmer = None):
        tokens = nltk.word_tokenize(text.lower().translate(self.punctuations))
        if stemmer is None:
            return tokens
        return [stemmer.stem(item) for item in tokens]

    def tfidf_vectorizer(self):
        # stemmer = nltk.stem.porter.PorterStemmer()
        # normalize = partial(self.text_norm, stemmer=stemmer)
        normalize = partial(self.text_norm, stemmer=None)
        return TfidfVectorizer(tokenizer=normalize, stop_words='english')

    def cosine_sim(self, text1, text2):
        """
        Calculate similarity of two texts

        :param text1: left text
        :param text2: right text
        :return: cosine similarity of two text vector
        """
        tfidf = self.vectorizer.fit_transform([text1, text2])
        return (tfidf * tfidf.T).A[0, 1]

    def bing_search(self, api_key, query, decoration, format, count, **kwargs):

        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {"q": query, "textDecorations": decoration, "textFormat": format, "count": count}

        response = requests.get(self.BING_API, headers=headers, params=params)
        if response.status_code == requests.status_codes.codes.OKAY:
            data = response.json()
            data = pd.DataFrame([(html.unescape(row['name']), row['url']) for row in data['webPages']['value']],
                                columns=['entity', 'url'])

            data['clean_entity'] = data['entity'].apply(self.__clean_text)
            return data.drop_duplicates(subset=['clean_entity']).drop(columns=['clean_entity'], axis=1).reset_index(
                drop=True)
        else:
            return pd.DataFrame()

    def is_loaded(self):
        return self.word2vec is not None

    @staticmethod
    def clean_payload(payload: EntityPayload):
        query = payload.query.encode("ascii", "ignore").decode()
        query = ' '.join([re.sub("[-_]+", "", line.strip()) for line in query.splitlines() if line])
        payload.query = query
        return payload

    @classmethod
    def pre_process(cls, payload: BaseModel):
        logger.debug("Pre-processing payload.")

        # generate hash uuid
        if isinstance(payload, EntityPayload):
            if payload.query is None:
                return None, None

            query = payload.query.encode("ascii", "ignore").decode()
            clean_text = ' '.join([re.sub("[-_]+", "", line.strip()) for line in query.splitlines() if line])

            # TODO: need to review, add to make sure to be the same with vizrefra
            # size = sys.getsizeof(clean_text)
            # logger.info(f"Input text size: {size}")
            # if size > self.MAX_TEXT_SIZE:
            #     # size bigger than 200kb
            #     # truncate
            #     tokens = clean_text.split()
            #     limit = self.MAX_TEXT_SIZE * len(tokens) // size // 2
            #     text = " ".join(tokens[:limit])
            # else:
            #     text = clean_text

            # clean_text = self.__clean_text(payload.query)

            return clean_text, helper.create_uuid(clean_text)
        elif isinstance(payload, BingSearchPayload):
            clean_text = payload.query.lower()
            return clean_text, helper.create_uuid(clean_text)
        elif isinstance(payload, TwitterTrendPayload):
            return "twitter", helper.create_interval_uuid()
        elif isinstance(payload, GoogleNewsPayload):
            return "google_news", helper.create_interval_uuid()
        else:
            return None, None

    @staticmethod
    def _post_process(uid: str, entities: pd.DataFrame, circles: pd.DataFrame, distance_matrix: pd.DataFrame,
                      tags: pd.DataFrame) -> EntityResult:
        logger.debug("Post-processing extraction.")

        return EntityResult(
            raw=RawEntityResult(uuid=uid, entities=entities.to_dict('records'), tags=tags.to_dict('records')),
            circles=RefineEntityResult(uuid=uid, entities=circles.to_dict('records')),
            distances=DistanceMatrixResult(uuid=uid,  entities=circles.to_dict('records'),
                                           distances=distance_matrix.values.tolist())
        )

    def create_circles(self, clean_query: str, entities: pd.DataFrame, max_entities: int = 50):
        # entities as one article
        entities = entities.head(max_entities).reset_index(drop=True).copy(deep=True)
        clean_text = self.__clean_text(clean_query)
        clean_entities = entities['entity'].apply(self.__clean_text)

        # cosine similarity
        tfidf_model = self.vectorizer.fit(clean_entities)
        entity_vector = tfidf_model.transform(clean_entities)

        # entity_article_sim = clean_entities.apply(lambda entity: self.cosine_sim(entity, article)).values
        # entity_text_sim = clean_entities.apply(lambda entity: self.cosine_sim(entity, clean_text)).values
        # entity_topk_sim = clean_entities.apply(lambda entity: self.cosine_sim(entity, topk)).values

        # scale the sim score
        # radius = entities[['entity']].copy()
        # radius = radius.set_index('entity')
        if 'salience' in entities:
            # radius['size'] = entities['sim']/(entities['sim'].iloc[0]/entity_text_sim[0])
            # radius['height'] = entity_text_sim
            text_vector = tfidf_model.transform([clean_text])
            entity_text_sim = cosine_similarity(entity_vector, text_vector).ravel()

            size = entities['salience'] / (entities['salience'].iloc[0] / entity_text_sim[0])
            height = entity_text_sim
        elif clean_text == 'twitter':
            height = entities['tweet_volume']
            size = height * 0.0000001
        elif clean_text == 'google_news':
            article = " ".join(clean_entities.astype(str).tolist())
            article_vector = tfidf_model.transform([article])
            entity_text_sim = cosine_similarity(entity_vector, article_vector).ravel()

            height = abs(entities['polarity'])
            size = entity_text_sim * .1
        else:
            # BING search
            # pick top five scored entities
            topk = " ".join(clean_entities.iloc[:5].astype(str).tolist())

            # size = entity_article_sim
            # height = entity_text_sim
            topk_vector = tfidf_model.transform([topk])
            article = " ".join(clean_entities.astype(str).tolist())
            article_vector = tfidf_model.transform([article])
            entity_article_sim = cosine_similarity(entity_vector, article_vector).ravel()
            entity_topk_sim = cosine_similarity(entity_vector, topk_vector).ravel()

            size = np.power(entity_article_sim, 2)
            size = (size - size.min()) / (size.max() - size.min())
            height = entity_topk_sim

        pairwise = cdist(clean_entities.values[:, np.newaxis], clean_entities.values[:, np.newaxis],
                         lambda u, v: self.word2vec.wmdistance(u[0].split(), v[0].split()))
        pairwise = pd.DataFrame(pairwise)
        pairwise = pairwise.replace([np.inf, -np.inf], np.nan).dropna(how='all', axis=1).dropna(how='all', axis=0)
        pairwise = pairwise.fillna(0)

        # run PCA
        if len(pairwise) <= 2:
            return pd.DataFrame(), pd.DataFrame()

        scaled_pairwise = self.scaler.fit_transform(pairwise)
        pca = PCA(n_components=2)
        pca.fit(scaled_pairwise)

        pca_matrix = np.vstack(
            (entities['entity'][pairwise.index].values, size[pairwise.index], height[pairwise.index], pca.components_))
        # vstack auto convert other numeric columns to object to fit with entity column
        pca_matrix = pd.DataFrame(pca_matrix, index=['entity', 'size', 'height', 'x', 'y']).transpose()

        # # keep meta columns
        # if 'tag' in entities.columns:
        #     key_cols = ['entity', 'tag']
        # else:
        #     key_cols = ['entity']
        meta_cols = ['entity'] + [col for col in entities.columns.difference(pca_matrix.columns)]
        pca_matrix = pca_matrix.merge(entities[meta_cols], on='entity', how='left')

        # clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1)
        pca_matrix['cluster'] = clusterer.fit_predict(scaled_pairwise)
        # pca_matrix = pca_matrix.sort_values('salience', ascending=False).groupby('cluster').apply(lambda x: x.to_dict('records'))

        return pairwise, pca_matrix

    def bing_transform(self, payload: BingSearchPayload):
        # clean_text = payload.query.lower()
        clean_text, uuid_ = self.pre_process(payload)
        entities = self.bing_search(**payload.dict())
        distance_matrix, circles = self.create_circles(clean_text, entities)
        return uuid_, entities, circles, distance_matrix, pd.DataFrame()

    def entity_transform(self, payload: Union[EntityPayload, None], text_id: Union[str, None]):

        if not self.is_loaded():
            raise ValueError(NOT_FOUND_MODEL)

        clean_query, uuid_ = self.pre_process(payload)
        if uuid_ is None and text_id is None:
            raise ValueError("Must include text query in payload or text_id in query")

        uuid_ = text_id if text_id is not None else uuid_
        # entities, tags = self.google_nlp.extract_entities(query=clean_query, limit=payload.limit, salience=payload.salience)
        # distance_matrix, circles = self.create_circles(clean_query, entities)
        # return entities, circles, distance_matrix, tags

        # checking database
        endpoint = f"/entity/raw?limit={payload.limit}&salience={payload.max_salience}"
        item = self.db.get_item(uuid_, endpoint)
        if item is None and clean_query is None:
            raise ValueError("Must include text query in payload or text_id in query")
        elif item is None:
            # prediction: EntityResult = model.predict(block_data)
            # g_entities = self.google_entities(query=payload.query)
            entities, tags = self.google_nlp.extract_entities(query=clean_query, limit=payload.limit,
                                                              salience=payload.max_salience)

            response: RawEntityResult = RawEntityResult(uuid=uuid_, entities=entities.to_dict('records'), tags=tags.to_dict('records'))

            self.db.create_item(item={
                'id': uuid_,
                'endpoint': endpoint,
                'json': response.json()
            })
        else:
            response: RawEntityResult = RawEntityResult(**json.loads(item['json']))
            entities, tags = pd.json_normalize(response.entities), pd.json_normalize(response.tags)

        # checking database
        endpoint = f"/entity/distance?ignore_overlap={payload.ignore_overlap}&salience={payload.max_salience}&size={payload.max_entities}"
        item = self.db.get_item(uuid_, endpoint)
        if item is None:
            distance_matrix, circles = self.create_circles(clean_query, entities, max_entities=payload.max_entities)

            if payload.ignore_overlap:
                circles = helper.trim_overlap(circles, scale=0.06, weight=1)
            response: DistanceMatrixResult = DistanceMatrixResult(entities=circles.to_dict('records'),
                                                                  distances=distance_matrix.values.tolist())
            self.db.create_item(item={
                'id': uuid_,
                'endpoint': endpoint,
                'json': response.json()
            })
        else:
            response: DistanceMatrixResult = DistanceMatrixResult(**json.loads(item['json']))
            distance_matrix, circles = pd.json_normalize(response.distances), pd.json_normalize(response.entities)

        logger.debug(f"entities: {entities.shape}")
        logger.debug(f"distance_matrix: {distance_matrix.shape}")
        logger.debug(f"circles: {circles.shape}")

        return uuid_, entities, circles, distance_matrix, tags

    def twitter_transform(self, payload: TwitterTrendPayload):
        trends = self.twitter_trend.get_trends(_id=payload.topic_id)
        if trends.empty:
            return ValueError(NO_VALID_PAYLOAD.format(payload))

        trends = trends.rename(columns={'name': 'entity'})

        distance_matrix, circles = self.create_circles('twitter', trends)
        if payload.ignore_overlap:
            circles = helper.trim_overlap(circles, scale=0.06, weight=1)

        return str(uuid.uuid4()), trends, circles, distance_matrix, pd.DataFrame()

    def google_news_transform(self, payload: GoogleNewsPayload):
        response = gnewsclient.NewsClient(language=payload.language, location=payload.location,
                                          topic=payload.topic, max_results=payload.max_results).get_news()
        news = pd.json_normalize(response)

        # get sentiment
        sentiment = news['title'].map(lambda text: self.google_nlp.extract_sentiment(text))
        news['polarity'] = sentiment.apply(lambda x: x.score)
        news['sentiment'] = sentiment.apply(lambda x: x.label)

        # remove duplicates
        news = news.sort_values(by='polarity', ascending=False)
        news = news.drop_duplicates(subset='title', keep="first").reset_index(drop=True)

        news = news.rename(columns={'title': 'entity', 'link': 'url'})
        distance_matrix, circles = self.create_circles('google_news', news)
        if payload.ignore_overlap:
            circles = helper.trim_overlap(circles, scale=0.06, weight=1)

        return str(uuid.uuid4()), news, circles, distance_matrix, pd.DataFrame()

    def predict(self, payload: BaseModel, text_id: str = None):
        if payload is None:
            raise ValueError(NO_VALID_PAYLOAD.format(payload))

        # clean_text, _ = self.pre_process(payload)

        # generate hash uuid
        if isinstance(payload, EntityPayload):
            prediction = self.entity_transform(payload, text_id)
        elif isinstance(payload, BingSearchPayload):
            prediction = self.bing_transform(payload)
        elif isinstance(payload, TwitterTrendPayload):
            prediction = self.twitter_transform(payload)
        elif isinstance(payload, GoogleNewsPayload):
            prediction = self.google_news_transform(payload)
        else:
            raise ValueError(NO_VALID_PAYLOAD.format(payload))

        # logger.info(prediction)
        post_processed_result = self._post_process(*prediction)

        return post_processed_result
