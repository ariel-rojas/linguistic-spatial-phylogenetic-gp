"""Contrato de componentes y selección de columnas para kernels sklearn."""

from abc import ABC, abstractmethod

from sklearn.gaussian_process.kernels import Kernel


class KernelComponent(ABC):
    """Estrategia: cada componente construye un kernel sklearn intercambiable."""

    @abstractmethod
    def build(self):
        """Devuelve la covarianza del componente, incluida su amplitud."""


class ColumnKernel(Kernel):
    """Aplica un kernel a X[:, start:stop], conservando la interfaz de sklearn.

    Es necesario porque Matérn debe ver la geografía y la filogenia el índice de hoja.
    theta, bounds y gradientes se delegan al kernel, sin implementar optimización.
    Los parámetros del constructor se conservan intactos para permitir clone().
    """

    def __init__(self, kernel, start=0, stop=None):
        self.kernel = kernel
        self.start = start
        self.stop = stop

    @property
    def theta(self):
        """Hiperparámetros del kernel interno en la representación de sklearn."""
        return self.kernel.theta

    @theta.setter
    def theta(self, value):
        self.kernel.theta = value

    @property
    def bounds(self):
        """Límites internos; vacíos en el experimento actual con valores fijos."""
        return self.kernel.bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        """Calcula covarianza (y opcionalmente gradiente) sobre las columnas elegidas."""
        X_selected = X[:, self.start : self.stop]
        Y_selected = None if Y is None else Y[:, self.start : self.stop]
        return self.kernel(X_selected, Y_selected, eval_gradient=eval_gradient)

    def diag(self, X):
        """Diagonal sin construir la matriz completa."""
        return self.kernel.diag(X[:, self.start : self.stop])

    def is_stationary(self):
        """Conserva la propiedad de estacionariedad del kernel interno."""
        return self.kernel.is_stationary()
