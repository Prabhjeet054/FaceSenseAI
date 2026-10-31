import numpy as np

from app.core.embedder import FaceEmbedder


class FakeArcFaceModel:
    """Deterministic ArcFace-shaped stub for tests without model downloads."""

    def get_feat(self, faces: list[np.ndarray]) -> np.ndarray:
        embeddings = []
        for face in faces:
            channel_means = face.astype(np.float32).mean(axis=(0, 1))
            embedding = np.zeros(512, dtype=np.float32)
            embedding[:3] = channel_means
            embeddings.append(embedding)
        return np.asarray(embeddings)


def test_embedding_has_512_dimensions_and_unit_norm() -> None:
    face = np.full((112, 112, 3), 128, dtype=np.uint8)
    embedding = FaceEmbedder(model=FakeArcFaceModel()).embed(face)

    assert embedding.shape == (512,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


def test_same_person_photos_are_more_similar_than_different_people() -> None:
    person_a_photo_1 = np.full((112, 112, 3), [220, 30, 30], dtype=np.uint8)
    person_a_photo_2 = np.full((112, 112, 3), [215, 35, 30], dtype=np.uint8)
    person_b_photo = np.full((112, 112, 3), [30, 30, 220], dtype=np.uint8)
    embedder = FaceEmbedder(model=FakeArcFaceModel())

    person_a_embedding_1 = embedder.embed(person_a_photo_1)
    person_a_embedding_2 = embedder.embed(person_a_photo_2)
    person_b_embedding = embedder.embed(person_b_photo)
    same_person_similarity = float(
        np.dot(person_a_embedding_1, person_a_embedding_2)
    )
    different_person_similarity = float(
        np.dot(person_a_embedding_1, person_b_embedding)
    )

    assert same_person_similarity > different_person_similarity
