import cadquery as cq

sketch = (
cq.Workplane("XY")
.sketch()
.rect(4, 2)
)
new_location = cq.Location(cq.Vector(5, 5, 0))
result = (
sketch.located(new_location)
)
cq.exporters.export(result, 'GT.stl')
cq.exporters.export(result, 'GT.stl')