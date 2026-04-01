import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 30, 10)
.faces(">Z")
.workplane()
.center(10, 10)
.cboreHole(5, 10, 4)
)
cq.exporters.export(result, 'GT.stl')