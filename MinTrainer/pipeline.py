
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor

from MinTrainer.transformers.distance_transformer import DistanceTransformer


def create_pipeline():

    # create the estimator
    model = RandomForestRegressor()
    model.set_params(n_estimators=100, max_depth=1)

    # create the pipeline
    pipe_distance = make_pipeline(
        DistanceTransformer(),
        StandardScaler())

    cols = ["pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude"]

    feateng_blocks = [
        ('distance', pipe_distance, cols),
    ]

    features_encoder = ColumnTransformer(feateng_blocks)

    pipeline = Pipeline(steps=[
                ('features', features_encoder),
                ('model', model)])

    return pipeline
