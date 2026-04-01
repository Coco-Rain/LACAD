import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 2)
.faces(">Z")
.workplane()
.center(3, 0)
.placeSketch(cq.Sketch().circle(1))
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')