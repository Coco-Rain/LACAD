import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(5, 25)
.extrude(8)
.faces(">Z")
.workplane()
.transformed(rotate=(0, 0, 45))
.moveTo(15, 0)
.radiusArc((0, 15), 20)
.close()
.cutBlind(-4)
.faces("<Z")
.workplane()
.cboreHole(6, 12, 5)
)
cq.exporters.export(result, 'GT.stl')