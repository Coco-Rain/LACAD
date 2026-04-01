import cadquery as cq

result = (
cq.Workplane("XY")
.rect(12, 8)
.extrude(1)
.faces(">Z")
.workplane()
.moveTo(-4, 0)
.lineTo(0, 0)
.threePointArc((2, 2), (4, 0))
.close()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')