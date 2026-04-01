import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 10)
.extrude(5)
.faces(">Z")
.workplane()
.rarray(4, 4, 3, 2)
.circle(1)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')