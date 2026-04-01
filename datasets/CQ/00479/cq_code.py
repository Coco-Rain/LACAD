import cadquery as cq

initial_sketch = (
cq.Sketch()
.rect(5, 5)
.circle(3)
)
new_location = cq.Location(cq.Vector(10, 10, 0))
moved_sketch = initial_sketch.moved(new_location)
result = (
cq.Workplane("XY")
.placeSketch(moved_sketch)
.extrude(1)
)
cq.exporters.export(result, 'GT.stl')