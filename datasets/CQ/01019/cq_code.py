import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.extrude(10)
.faces("<Z")
.workplane(invert=True)
.polygon(6, 6.35)
.cutBlind(3)
)
cq.exporters.export(result, 'GT.stl')