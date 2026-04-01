import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 15, 8)
.faces(">Z")
.shells()
.shell(-2)
.faces(">Z")
.workplane()
.circle(5)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')