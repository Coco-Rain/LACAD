import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.lineTo(20, 0)
.threePointArc((25, 5), (20, 10))
.lineTo(0, 10)
.close()
.extrude(8)
.faces(">Z")
.workplane()
.cboreHole(6, 12, 5)
)
cq.exporters.export(result, 'GT.stl')