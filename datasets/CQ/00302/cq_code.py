import cadquery as cq

sketch = cq.Sketch().rect(2, 2)
result = (
cq.Workplane("XY")
.box(10, 10, 1)
.faces(">Z")
.workplane()
.placeSketch(sketch)
)
cq.exporters.export(result, 'GT.stl')