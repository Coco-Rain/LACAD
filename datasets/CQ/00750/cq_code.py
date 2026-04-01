import cadquery as cq

sketch = cq.Sketch().circle(3)
new_location = cq.Location(cq.Vector(10, 10, 0))
result = (
sketch.located(new_location)
)
cq.exporters.export(result, 'GT.stl')