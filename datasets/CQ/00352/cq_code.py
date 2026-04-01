import cadquery as cq

sketch = cq.Sketch().circle(0.8)
result = (
cq.Workplane("XY")
.box(20, 20, 5)
.faces(">Z")
.workplane()
.rect(16, 16, forConstruction=True)
.vertices()
.placeSketch(sketch)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')