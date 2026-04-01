import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 10)
.extrude(5)
.faces(">Z")
.workplane()
.center(-5, -2)
.slot2D(15, 3)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')