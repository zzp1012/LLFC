import torch

class DissimilarityMetric:
    """compute the distance of two featuremaps
    """
    def __init__(self, metric):
        self.__metric = metric

    def __call__(self, A, B, **kwargs):
        if self.__metric == "vanilla":
            return self.__vanilla(A, B)
        elif self.__metric == "cosine":
            return self.__cosine_similarity(A, B, **kwargs)
        else:
            raise ValueError("Unknown metric")

    def __vanilla(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """compute the vanilla distance between two featuremaps, Frobenius norm

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
        """
        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"
        A = A.view(A.shape[0], -1) # shape: (N, D)
        B = B.view(B.shape[0], -1) # shape: (N, D)

        def norm_square(A: torch.Tensor) -> torch.Tensor:
            return torch.sum(A ** 2) # shape: (1, )

        return norm_square(A - B) / torch.norm(A, p="fro") / torch.norm(B, p="fro")


    def __cosine_similarity(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute the cosine similarity between two matrices

        dist = 1 - <A, B> / (||A|| * ||B||)

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
            kwargs: the keyword arguments

        Return:
            dist (torch.Tensor): the distance between A and B
        """
        get_coef = kwargs.get("get_coef", False)

        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"

        A = A.view(A.shape[0], -1).double() # shape: (N, D)
        B = B.view(B.shape[0], -1).double() # shape: (N, D)

        # compute the frobenius inner product of A and B
        inner_product = torch.sum(A * B) # shape: (1, 1)
        # compute the frobenius norm of A and B
        A_norm = torch.norm(A, p="fro") # shape: (1, 1)
        B_norm = torch.norm(B, p="fro") # shape: (1, 1)

        # cal the distance
        dist = 1 - torch.abs(inner_product) / (A_norm * B_norm)

        # compute the coefficient
        coef = inner_product / (B_norm ** 2)

        assert torch.abs(inner_product) <= A_norm * B_norm * (1 + 1e-10), \
            f"the inner product - {inner_product} should be less than the product of the norm - {A_norm * B_norm}"

        if get_coef:
            return dist, coef
        else:
            return dist


class DissimilarityMetricOverSamples:
    """compute the distance of two featuremaps, for each sample
    """
    def __init__(self, metric):
        self.__metric = metric

    def __call__(self, A, B, **kwargs):
        if self.__metric == "vanilla":
            return self.__vanilla(A, B)
        elif self.__metric == "cosine":
            return self.__cosine_similarity(A, B, **kwargs)

    def __vanilla(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """compute the vanilla distance between two featuremaps, Frobenius norm

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
        """
        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"
        A = A.view(A.shape[0], -1) # shape: (N, D)
        B = B.view(B.shape[0], -1) # shape: (N, D)

        def norm_square(A: torch.Tensor) -> torch.Tensor:
            return torch.sum(A ** 2, dim=-1) # shape: (N, )
        return norm_square(A - B) / torch.norm(A, p="fro", dim=-1) / torch.norm(B, p="fro", dim=-1) # shape: (N, )


    def __cosine_similarity(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute the cosine similarity between two matrices

        dist = 1 - <A, B> / (||A|| * ||B||)

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
            kwargs: the keyword arguments

        Return:
            dist (torch.Tensor): the distance between A and B
        """
        get_coef = kwargs.get("get_coef", False)

        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"

        A = A.view(A.shape[0], -1).double() # shape: (N, D)
        B = B.view(B.shape[0], -1).double() # shape: (N, D)

        # compute the frobenius inner product of A and B
        inner_product = torch.sum(A * B, dim=-1) # shape: (N, )
        # compute the frobenius norm of A and B
        A_norm = torch.norm(A, p="fro", dim=-1) # shape: (N, )
        B_norm = torch.norm(B, p="fro", dim=-1) # shape: (N, )

        # cal the distance
        dist = 1 - torch.abs(inner_product) / (A_norm * B_norm) # shape: (N, )

        # compute the coefficient
        coef = inner_product / (B_norm ** 2) # shape: (N, )

        if get_coef:
            return dist, coef
        else:
            return dist
