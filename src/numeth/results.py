import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class NumericalResult(float):
    """Result class for methods returning a single float value."""
    def __new__(cls, value, **kwargs):
        return super().__new__(cls, value)

    def __init__(self, value, method_info):
        self.method_info = method_info

    def graph(self):
        if plt is None:
            print("Matplotlib is required for visualization.")
            return
        
        m_type = self.method_info.get('type')
        if m_type == 'integration':
            self._graph_integration()
        elif m_type == 'differentiation':
            self._graph_differentiation()
        elif m_type == 'interpolation':
            self._graph_interpolation()
        else:
            print(f"Visualization not implemented for type: {m_type}")

    def _graph_integration(self):
        f = self.method_info['f']
        a = self.method_info['a']
        b = self.method_info['b']
        method = self.method_info['method']
        
        x = np.linspace(a - 0.5, b + 0.5, 400)
        y = [f(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b', label='f(x)')
        
        # Integration specific visualization
        ix = np.linspace(a, b, 100)
        iy = [f(ixi) for ixi in ix]
        plt.fill_between(ix, iy, alpha=0.3, label=f'Integral ({method})')
        
        plt.axhline(0, color='black', lw=1)
        plt.title(f"Integration Visualization: {method}")
        plt.legend()
        plt.show()

    def _graph_differentiation(self):
        f = self.method_info['f']
        x_pt = self.method_info['x']
        h = self.method_info['h']
        val = float(self)
        
        x = np.linspace(x_pt - 2*h, x_pt + 2*h, 400)
        y = [f(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='f(x)')
        
        # Tangent line
        y_pt = f(x_pt)
        tangent = y_pt + val * (x - x_pt)
        plt.plot(x, tangent, '--r', label="Approximated Tangent")
        plt.plot(x_pt, y_pt, 'ro')
        
        plt.title(f"Differentiation Visualization: {self.method_info['method']}")
        plt.legend()
        plt.show()

    def _graph_interpolation(self):
        points = self.method_info['points']
        f_interp = self.method_info['f_interp']
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        x_plot = np.linspace(min(xs) - 0.5, max(xs) + 0.5, 400)
        y_plot = [f_interp(xi) for xi in x_plot]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, label='Interpolated Curve')
        plt.scatter(xs, ys, color='red', label='Data Points')
        
        plt.title(f"Interpolation Visualization: {self.method_info['method']}")
        plt.legend()
        plt.show()

class IterativeResult(tuple):
    """Result class for methods returning (value, iterations, success)."""
    def __new__(cls, *args, method_info=None):
        return super().__new__(cls, args)

    def __init__(self, *args, method_info=None):
        self.method_info = method_info

    def graph(self):
        if plt is None:
            print("Matplotlib is required for visualization.")
            return
            
        m_type = self.method_info.get('type')
        if m_type == 'root_finding':
            self._graph_root_finding()
        elif m_type == 'optimization':
            self._graph_optimization()
        else:
            print(f"Visualization not implemented for type: {m_type}")

    def _graph_root_finding(self):
        f = self.method_info['f']
        root = self[0]
        method = self.method_info['method']
        
        # Try to find a reasonable range
        a = self.method_info.get('a', root - 1)
        b = self.method_info.get('b', root + 1)
        
        x = np.linspace(a - 0.5, b + 0.5, 400)
        y = [f(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='f(x)')
        plt.axhline(0, color='black', lw=1)
        plt.plot(root, 0, 'ro', label=f'Root: {root:.4f}')
        
        plt.title(f"Root Finding Visualization: {method}")
        plt.legend()
        plt.show()

    def _graph_optimization(self):
        f = self.method_info['f']
        x_min = self[0]
        y_min = self[1]
        method = self.method_info['method']
        
        a = self.method_info.get('a', x_min - 1)
        b = self.method_info.get('b', x_min + 1)
        
        x = np.linspace(a - 0.5, b + 0.5, 400)
        y = [f(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='f(x)')
        plt.plot(x_min, y_min, 'ro', label=f'Min: ({x_min:.4f}, {y_min:.4f})')
        
        plt.title(f"Optimization Visualization: {method}")
        plt.legend()
        plt.show()
