import cadquery as cq

result = (
cq.Sketch()
.rect(12, 6)
.ellipse(6, 4)
)
new_sketch = result.copy()
cq.exporters.export(result, 'GT.stl')