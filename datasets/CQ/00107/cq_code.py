import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(nSides=6, diameter=20)
.extrude(5)
.faces(">Z").workplane()
.transformed(offset=(0, 4, 3))
.radiusArc(endPoint=(4, 0), radius=6)
.close()
.extrude(2)
.faces(">Z[-2]").workplane()
.hole(3)
)
cq.exporters.export(result, 'GT.stl')