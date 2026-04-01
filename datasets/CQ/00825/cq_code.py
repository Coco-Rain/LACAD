import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 5)
.faces(">Z")
.workplane()
.center(5, 5)
.cboreHole(3, 7, 2)
)
cq.exporters.export(result, 'GT.stl')