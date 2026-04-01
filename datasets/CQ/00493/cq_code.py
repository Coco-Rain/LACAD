import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(2, 2)
.rect(6, 4)
.extrude(3)
.faces(">Z")
.workplane()
.center(-2, -1)
.circle(1)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')