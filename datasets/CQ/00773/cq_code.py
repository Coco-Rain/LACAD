import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.lineTo(15, 0)
.threePointArc((20, 5), (15, 10))
.lineTo(0, 10)
.close()
.extrude(6)
.faces(">Z")
.workplane()
.slot2D(12, 4)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')